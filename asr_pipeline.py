import os
import torch
import librosa
import gc
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

class ASREngine:
    def __init__(self, model_name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_name = None
        self.processor = None
        self.base_model = None
        self.current_model = None
        self.loaded_adapters = {} # {adapter_id: PeftModel}
        
        # Silero VAD
        self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils
        
        # Initial Load if specified
        if model_name:
            self.switch_base_model(model_name)

    def switch_base_model(self, new_model_name):
        if self.current_model_name == new_model_name:
            return
            
        print(f"Flushing VRAM and loading {new_model_name}...", flush=True)
        # Clear existing models from VRAM
        if self.base_model is not None:
            del self.base_model
            del self.current_model
            self.loaded_adapters = {}
            gc.collect()
            torch.cuda.empty_cache()
            
        self.current_model_name = new_model_name
        self.processor = WhisperProcessor.from_pretrained(new_model_name)
        
        # Use device_map="auto" to automatically offload layers to RAM if VRAM is full
        self.base_model = WhisperForConditionalGeneration.from_pretrained(
            new_model_name, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.base_model.eval()
        self.current_model = self.base_model


    def load_adapter(self, lang_code, adapter_path):
        adapter_id = f"{self.current_model_name}_{lang_code}"
        if adapter_id not in self.loaded_adapters:
            print(f"Loading Adapter: {adapter_path}...", flush=True)
            if not Path(adapter_path).exists():
                raise FileNotFoundError(f"Adapter not found at {adapter_path}")
            
            # Create a PeftModel from the base model
            self.loaded_adapters[adapter_id] = PeftModel.from_pretrained(
                self.base_model, 
                adapter_path, 
                adapter_name=adapter_id
            )
        
        self.current_model = self.loaded_adapters[adapter_id]
        self.current_model.set_adapter(adapter_id)
        self.current_model.eval()

    def use_base_model(self):
        self.current_model = self.base_model
        self.current_model.eval()

    def denoise_audio(self, audio):
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.8)

    def preprocess_audio(self, audio_path, apply_vad=True, apply_denoise=True):
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32, copy=False)
        
        if apply_vad:
            wav = torch.from_numpy(audio)
            speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, sampling_rate=16000)
            if speech_timestamps:
                segments = [audio[ts['start']:ts['end']] for ts in speech_timestamps]
                audio = np.concatenate(segments)

        if apply_denoise and len(audio) > 0:
            audio = self.denoise_audio(audio)
        
        audio = librosa.util.normalize(audio)
        return audio

    def transcribe_audio_array(self, audio, language_code, task="transcribe"):
        chunk_length_s = 30
        chunk_samples = chunk_length_s * 16000
        full_transcription = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            if len(chunk) < 1600: continue
            
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device, dtype=torch.float16)
            
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language_code, task=task)
            
            with torch.no_grad():
                predicted_ids = self.current_model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256
                )
            
            chunk_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if chunk_text.strip():
                full_transcription.append(chunk_text.strip())
            
        return " ".join(full_transcription).strip()

    def transcribe(self, audio, language_code, task="transcribe"):
        return self.transcribe_audio_array(audio, language_code, task=task)

    def postprocess_text(self, text, lang_code):
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang_code)
        return normalizer.normalize(text)

engine = None
def get_engine():
    global engine
    if engine is None:
        engine = ASREngine()
    return engine
