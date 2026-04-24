import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import evaluate

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:
    raise ImportError(
        "peft is not available or is broken in this environment. "
        "Please reinstall the requirements after fixing package versions."
    ) from exc

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


MODEL_NAME = "openai/whisper-small"
LANGUAGES = (
    {"code": "hi", "name": "hindi"},
    {"code": "mr", "name": "marathi"},
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper with QLoRA.")
    # Automatically resolve the base directory relative to this script
    script_dir = Path(__file__).parent.absolute()
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--output-dir", default=None, help="Override the output root directory.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--languages", nargs="+", default=["hi", "mr"], help="Language codes to train.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-label-length", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def build_language_map():
    return {entry["code"]: entry["name"] for entry in LANGUAGES}


def resolve_language(code):
    lang_map = build_language_map()
    if code not in lang_map:
        raise ValueError(f"Unsupported language code: {code}. Expected one of {sorted(lang_map)}")
    return lang_map[code]


def prepare_output_paths(base_dir, code, output_root=None):
    root = Path(output_root) if output_root else Path(base_dir) / "output"
    lang_dir = root / f"whisper-small-{code}-qlora"
    lang_dir.mkdir(parents=True, exist_ok=True)
    return lang_dir


def load_lang_dataset(base_dir, code):
    data_dir = Path(base_dir) / "data" / code
    train_path = data_dir / "train_metadata.jsonl"
    test_path = data_dir / "test_metadata.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing metadata files for {code}: {train_path} / {test_path}")

    dataset = load_dataset("json", data_files={"train": str(train_path), "test": str(test_path)})
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    return dataset


def build_processor_and_model(model_name, language_name):
    processor = WhisperProcessor.from_pretrained(model_name, language=language_name, task="transcribe")

    use_qlora = torch.cuda.is_available() and BitsAndBytesConfig is not None
    if use_qlora:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            use_qlora = False

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("QLoRA backend not available. Falling back to standard LoRA on full precision model.")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.apply_spec_augment = True
    model.config.use_cache = False
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_name, task="transcribe")
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = language_name
    model.generation_config.task = "transcribe"

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return processor, model


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    max_label_length: int = 256

    def __call__(self, features):
        inputs = [self.processor.feature_extractor(feature["audio_filepath"]["array"], sampling_rate=16000).input_features[0] for feature in features]
        batch = self.processor.feature_extractor.pad(
            [{"input_features": input_feature} for input_feature in inputs],
            return_tensors="pt",
        )

        label_ids = [self.processor.tokenizer(feature["text"], truncation=True, max_length=self.max_label_length).input_ids for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": ids} for ids in label_ids],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def compute_metrics_factory(processor):
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        if pred_ids.ndim == 3:
            pred_ids = torch.from_numpy(pred_ids).argmax(dim=-1).numpy()

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def build_generation_config(processor, language_name):
    return {
        "forced_decoder_ids": processor.get_decoder_prompt_ids(language=language_name, task="transcribe"),
        "language": language_name,
        "task": "transcribe",
    }


def save_training_artifacts(trainer, processor, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "training_logs.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(logs_path, index=False)
    processor.save_pretrained(output_dir / "processor")
    return logs_path


def save_sample_predictions(trainer, processor, output_dir, sample_count=5):
    sample_count = min(sample_count, len(trainer.eval_dataset))
    if sample_count <= 0:
        return
    sample = trainer.eval_dataset.select(range(sample_count))
    predictions = trainer.predict(sample)
    pred_ids = predictions.predictions
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(axis=-1)
    label_ids = predictions.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    rows = []
    for idx in range(sample_count):
        rows.append(
            {
                "sample_index": idx,
                "prediction": processor.tokenizer.decode(pred_ids[idx], skip_special_tokens=True),
                "reference": processor.tokenizer.decode(label_ids[idx], skip_special_tokens=True),
            }
        )
    pd.DataFrame(rows).to_csv(Path(output_dir) / "sample_predictions.csv", index=False)


def train_language(args, code):
    language_name = resolve_language(code)
    base_dir = Path(args.base_dir)
    output_dir = prepare_output_paths(base_dir, code, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"Training Whisper-Small QLoRA for {language_name.upper()} ({code})")
    print(f"{'=' * 60}")

    dataset = load_lang_dataset(base_dir, code)
    processor, model = build_processor_and_model(args.model_name, language_name)
    generation_config = build_generation_config(processor, language_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=WhisperDataCollator(processor=processor, max_label_length=args.max_label_length),
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_factory(processor),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    save_training_artifacts(trainer, processor, output_dir)
    save_sample_predictions(trainer, processor, output_dir)

    adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Saved adapter: {adapter_dir}")
    print(f"Saved logs: {output_dir / 'training_logs.csv'}")
    print(f"Saved eval metrics: {output_dir / 'eval_metrics.json'}")

    del trainer, model, processor, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    for lang_code in args.languages:
        train_language(args, lang_code)
    print("\nTraining complete for all requested languages.")


if __name__ == "__main__":
    main()
