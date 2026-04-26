import os
import json
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
import random
from pathlib import Path

# --- CONFIGURATION ---
# Using google/fleurs because it is much faster to stream for 5-hour subsets
DATASET_NAME = "google/fleurs" 
LANG_MAP = {"marathi": "mr_in", "hindi": "hi_in"} # FLEURS uses regional codes
FOLDER_MAP = {"marathi": "mr", "hindi": "hi"}

TARGET_HOURS = 10
TARGET_SECONDS = TARGET_HOURS * 3600
# Make paths relative to this script
script_dir = Path(__file__).parent.absolute()
BASE_DATA_DIR = script_dir / "data"
TRAIN_SPLIT = 0.9

def download_subset(lang_full):
    lang_config = LANG_MAP[lang_full]
    lang_short = FOLDER_MAP[lang_full]
    
    print(f"\n🚀 Starting collection for: {lang_full} ({lang_config})")
    lang_dir = os.path.join(BASE_DATA_DIR, lang_short)
    clips_dir = os.path.join(lang_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    print(f"📡 Connecting to Hugging Face for {DATASET_NAME}...")
    # Added trust_remote_code=True for FLEURS loading script
    ds = load_dataset(DATASET_NAME, lang_config, split="train", streaming=True, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    all_data = []
    current_seconds = 0
    
    print(f"📥 Downloading clips to {clips_dir}...")
    for i, example in enumerate(ds):
        if example.get("audio") is None:
            continue
            
        audio_array = example["audio"]["array"]
        # FLEURS uses 'transcription' or 'raw_transcription'
        sentence = example.get("transcription") or example.get("raw_transcription") or example.get("text")
        
        if audio_array is None or sentence is None:
            continue
            
        duration = len(audio_array) / 16000
        if current_seconds + duration > TARGET_SECONDS:
            break
            
        filename = f"{lang_short}_clip_{i}.wav"
        filepath = os.path.join(clips_dir, filename)
        
        if not os.path.exists(filepath):
            sf.write(filepath, audio_array, 16000)
            # Normal print instead of \r so you can see the history
            print(f"✅ Saved clip {i}: {duration:.1f}s | Total: {current_seconds/60:.2f} min")
        
        all_data.append({
            "audio_filepath": filepath,
            "text": sentence,
            "duration": duration
        })
        current_seconds += duration

    # --- SHUFFLE AND SPLIT ---
    random.shuffle(all_data)
    split_idx = int(len(all_data) * TRAIN_SPLIT)
    
    # Save Train Metadata
    with open(os.path.join(lang_dir, "train_metadata.jsonl"), "w", encoding="utf-8") as f:
        for entry in all_data[:split_idx]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save Test Metadata
    with open(os.path.join(lang_dir, "test_metadata.jsonl"), "w", encoding="utf-8") as f:
        for entry in all_data[split_idx:]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✨ Finished {lang_full}. Total: {current_seconds/3600:.2f} hours.")

if __name__ == "__main__":
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    random.seed(42)
    for lang in LANG_MAP.keys():
        download_subset(lang)
