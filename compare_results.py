import argparse
import json
import gc
from pathlib import Path

import librosa
import pandas as pd
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import evaluate
from jiwer import cer, mer, wer, wil

BASE_MODEL = "openai/whisper-medium"

def parse_args():
    parser = argparse.ArgumentParser(description="Compare base Whisper vs QLoRA adapter.")
    script_dir = Path(__file__).parent.absolute()
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--language", choices=["hi", "mr"], required=True)
    parser.add_argument("--adapter-path", default=None, help="Adapter directory.")
    parser.add_argument("--audio-file", default=None, help="Optional single audio file to compare.")
    parser.add_argument("--sample-count", type=int, default=None)
    return parser.parse_args()

def language_name(code):
    return "hindi" if code == "hi" else "marathi"

def normalize_text(text):
    return " ".join(str(text).split()).strip()

def transcribe_all(model_type, model_name, adapter_path, samples, language_code):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    
    print(f"📦 Loading {model_type} model in FP16...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if adapter_path:
        print(f"🎨 Attaching Adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    predictions = []
    
    print(f"🎙️ Transcribing {len(samples)} samples with {model_type}...")
    from tqdm import tqdm
    for sample in tqdm(samples, desc=f"Evaluating {model_type}"):
        audio_path = sample["audio_filepath"]["path"]
        audio, _ = librosa.load(audio_path, sr=16000)
        # Cast inputs to float16 to match the model's dtype
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float16)
        
        with torch.no_grad():
            predicted_ids = model.generate(inputs, language=language_name(language_code), task="transcribe")
        
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        predictions.append(text)
        
    # Cleanup memory
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return predictions

def main():
    args = parse_args()
    checkpoints = {"hi": "1500", "mr": "2000"}
    default_adapter = Path(args.base_dir) / "output" / f"whisper-medium-{args.language}-qlora" / f"checkpoint-{checkpoints[args.language]}"
    adapter_path = args.adapter_path or default_adapter
    
    # Load samples
    data_dir = Path(args.base_dir) / "data" / args.language
    test_path = data_dir / "test_metadata.jsonl"
    dataset = load_dataset("json", data_files={"test": str(test_path)})["test"]
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    
    if args.sample_count:
        dataset = dataset.select(range(min(args.sample_count, len(dataset))))
    
    samples = [s for s in dataset]
    refs = [s["text"] for s in samples]
    
    # Run Sequentially to save VRAM
    base_preds = transcribe_all("BASE", BASE_MODEL, None, samples, args.language)
    adapter_preds = transcribe_all("ADAPTER", BASE_MODEL, adapter_path, samples, args.language)
    
    # Compute Metrics
    print("📊 Computing Final Metrics...")
    rows = []
    for i in range(len(samples)):
        ref = normalize_text(refs[i])
        b_p = normalize_text(base_preds[i])
        a_p = normalize_text(adapter_preds[i])
        
        rows.append({
            "audio_path": samples[i]["audio_filepath"]["path"],
            "reference": refs[i],
            "base_prediction": base_preds[i],
            "adapter_prediction": adapter_preds[i],
            "base_wer": 100 * wer([ref], [b_p]),
            "adapter_wer": 100 * wer([ref], [a_p]),
            "base_cer": 100 * cer([ref], [b_p]),
            "adapter_cer": 100 * cer([ref], [a_p]),
        })
    
    # Save Results
    output_dir = Path(adapter_path).parent
    pd.DataFrame(rows).to_csv(output_dir / "comparison_results.csv", index=False)
    
    summary = {
        "base_wer": 100 * wer([normalize_text(r) for r in refs], [normalize_text(p) for p in base_preds]),
        "adapter_wer": 100 * wer([normalize_text(r) for r in refs], [normalize_text(p) for p in adapter_preds]),
        "base_cer": 100 * cer([normalize_text(r) for r in refs], [normalize_text(p) for p in base_preds]),
        "adapter_cer": 100 * cer([normalize_text(r) for r in refs], [normalize_text(p) for p in adapter_preds]),
        "samples": len(refs)
    }
    
    with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✅ Done! Summary:\nBase WER: {summary['base_wer']:.2f}% | Adapter WER: {summary['adapter_wer']:.2f}%")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
