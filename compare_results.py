import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import evaluate
from jiwer import cer, mer, wer, wil


BASE_MODEL = "openai/whisper-small"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base Whisper vs QLoRA adapter.")
    script_dir = Path(__file__).parent.absolute()
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--language", choices=["hi", "mr"], required=True)
    parser.add_argument("--adapter-path", default=None, help="Adapter directory. Defaults to output/whisper-small-<lang>-qlora/final_adapter")
    parser.add_argument("--audio-file", default=None, help="Optional single audio file to compare.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional limit on test samples. Default is the full test split.",
    )
    parser.add_argument("--output-file", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def language_name(code):
    return {"hi": "hindi", "mr": "marathi"}[code]


def load_models(base_model_name, adapter_path, language):
    processor = WhisperProcessor.from_pretrained(base_model_name, language=language_name(language), task="transcribe")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)
    base_model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_name(language), task="transcribe")
    base_model.generation_config.language = language_name(language)
    base_model.generation_config.task = "transcribe"
    base_model.eval()

    adapter_base = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path).to(device)
    adapter_model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_name(language), task="transcribe")
    adapter_model.generation_config.language = language_name(language)
    adapter_model.generation_config.task = "transcribe"
    adapter_model.eval()
    return processor, base_model, adapter_model


def transcribe_audio(model, processor, audio_path, language_code):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, language=language_name(language_code), task="transcribe")
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def load_eval_samples(base_dir, language, sample_count):
    data_dir = Path(base_dir) / "data" / language
    test_path = data_dir / "test_metadata.jsonl"
    dataset = load_dataset("json", data_files={"test": str(test_path)})["test"]
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    if sample_count is None:
        return dataset
    return dataset.select(range(min(sample_count, len(dataset))))


def normalize_text(text):
    return " ".join(str(text).split()).strip()


def summarize_predictions(base_preds, adapter_preds, refs):
    base_refs = [normalize_text(x) for x in refs]
    base_preds = [normalize_text(x) for x in base_preds]
    adapter_preds = [normalize_text(x) for x in adapter_preds]

    metrics = {
        "base_wer": 100 * wer(base_refs, base_preds),
        "adapter_wer": 100 * wer(base_refs, adapter_preds),
        "base_cer": 100 * cer(base_refs, base_preds),
        "adapter_cer": 100 * cer(base_refs, adapter_preds),
        "base_mer": 100 * mer(base_refs, base_preds),
        "adapter_mer": 100 * mer(base_refs, adapter_preds),
        "base_wil": 100 * wil(base_refs, base_preds),
        "adapter_wil": 100 * wil(base_refs, adapter_preds),
        "exact_match_base": 100 * sum(p == r for p, r in zip(base_preds, base_refs)) / max(len(refs), 1),
        "exact_match_adapter": 100 * sum(p == r for p, r in zip(adapter_preds, base_refs)) / max(len(refs), 1),
        "samples": len(refs),
    }
    return metrics


def evaluate_samples(processor, base_model, adapter_model, samples, language_code):
    rows = []
    base_preds = []
    adapter_preds = []
    refs = []

    for sample in samples:
        audio_path = sample["audio_filepath"]["path"]
        reference = sample["text"]
        base_text = transcribe_audio(base_model, processor, audio_path, language_code)
        adapter_text = transcribe_audio(adapter_model, processor, audio_path, language_code)
        rows.append(
            {
                "audio_path": audio_path,
                "reference": reference,
                "base_prediction": base_text,
                "adapter_prediction": adapter_text,
                "base_wer": 100 * wer([normalize_text(reference)], [normalize_text(base_text)]),
                "adapter_wer": 100 * wer([normalize_text(reference)], [normalize_text(adapter_text)]),
                "base_cer": 100 * cer([normalize_text(reference)], [normalize_text(base_text)]),
                "adapter_cer": 100 * cer([normalize_text(reference)], [normalize_text(adapter_text)]),
            }
        )
        base_preds.append(base_text)
        adapter_preds.append(adapter_text)
        refs.append(reference)

    summary = summarize_predictions(base_preds, adapter_preds, refs)
    return rows, summary


def compare_single_file(processor, base_model, adapter_model, audio_file, language_code):
    base_text = transcribe_audio(base_model, processor, audio_file, language_code)
    adapter_text = transcribe_audio(adapter_model, processor, audio_file, language_code)
    rows = [
        {
            "audio_path": audio_file,
            "reference": "",
            "base_prediction": base_text,
            "adapter_prediction": adapter_text,
            "base_wer": None,
            "adapter_wer": None,
            "base_cer": None,
            "adapter_cer": None,
        }
    ]
    return rows, {"samples": 1}


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / max(len(values), 1)


def main():
    args = parse_args()
    adapter_path = args.adapter_path or Path(args.base_dir) / "output" / f"whisper-small-{args.language}-qlora" / "final_adapter"
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    processor, base_model, adapter_model = load_models(BASE_MODEL, str(adapter_path), args.language)

    if args.audio_file:
        rows, summary = compare_single_file(processor, base_model, adapter_model, args.audio_file, args.language)
    else:
        samples = load_eval_samples(args.base_dir, args.language, args.sample_count)
        rows, summary = evaluate_samples(processor, base_model, adapter_model, samples, args.language)

    df = pd.DataFrame(rows)
    output_file = Path(args.output_file) if args.output_file else adapter_path.parent / "comparison_results.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    summary_path = output_file.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(df.to_string(index=False))
    print(f"\nSaved comparison CSV: {output_file}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
