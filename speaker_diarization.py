import gc
import os
import matplotlib
matplotlib.use('Agg')
import torch
import librosa
import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

_PIPELINE = None
_PIPELINE_TOKEN = None
_FORCE_RELOAD = True

def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hf_token(explicit_token: Optional[str] = None) -> str:
    token = explicit_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.")
    return token

def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)

def merge_adjacent_turns(turns: List[Dict[str, float]], max_gap_seconds: float = 0.5) -> List[Dict[str, float]]:
    if not turns: return []
    ordered = sorted(turns, key=lambda item: (item["start"], item["end"]))
    merged = [dict(ordered[0])]
    for turn in ordered[1:]:
        prev = merged[-1]
        gap = turn["start"] - prev["end"]
        if turn["speaker"] == prev["speaker"] and gap <= max_gap_seconds:
            prev["end"] = max(prev["end"], turn["end"])
            continue
        merged.append(dict(turn))
    return merged

def format_diarized_transcript(segments: List[Dict[str, str]]) -> str:
    lines = []
    for segment in segments:
        lines.append(f"Speaker: {segment['speaker']}")
        lines.append(f"Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
        lines.append(segment['text'].strip())
        lines.append("-" * 54)
    return "\n".join(lines).strip()

def transcribe_diarized_audio(
    audio_path: str,
    engine,
    language_code: str,
    hf_token: Optional[str] = None,
    min_speakers: int = 2,
    max_speakers: int = 2,
    task: str = "transcribe",
    max_gap_seconds: float = 0.5,
    min_segment_seconds: float = 0.40,
    speaker_role_map: Optional[Dict[str, str]] = None,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
):
    token = get_hf_token(hf_token)
    
    # 1. Load Audio
    if audio is None:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    else:
        sr = sample_rate
        audio = np.asarray(audio, dtype=np.float32)
    audio = _normalize_audio(audio)

    # 2. SURGICAL Boundary Detection with Silero VAD
    # This is much more accurate for Marathi/Indic languages than Pyannote's internal VAD
    wav = torch.from_numpy(audio)
    speech_timestamps = engine.get_speech_timestamps(wav, engine.vad_model, sampling_rate=16000)
    
    if not speech_timestamps:
        return {"segments": [], "transcript": "No speech detected.", "plot_path": None}

    # 3. Speaker Embedding Extraction
    # We use a dedicated embedding model which is language-agnostic
    from pyannote.audio import Model
    from pyannote.audio.pipelines.utils import get_model
    
    print("Extracting speaker embeddings for segments...", flush=True)
    try:
        # Try to load the dedicated embedding model
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    except:
        # Fallback to the diarization model's internal encoder
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        model = pipeline._model # This is usually the segmentation/embedding model
    
    model.to(_resolve_device())
    model.eval()

    embeddings = []
    valid_turns = []
    
    for ts in speech_timestamps:
        start_idx = ts['start']
        end_idx = ts['end']
        duration = (end_idx - start_idx) / 16000
        
        if duration < 0.5: continue # Ignore very short blips
        
        # Extract chunk and convert to tensor
        chunk = audio[start_idx:end_idx]
        # Pyannote embedding model expects (batch, channel, samples)
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(_resolve_device())
        
        with torch.no_grad():
            try:
                # Get the embedding (usually a 512-dim vector)
                emb = model(chunk_tensor).cpu().numpy().flatten()
                embeddings.append(emb)
                valid_turns.append({"start": start_idx/16000, "end": end_idx/16000})
            except Exception as e:
                print(f"Skipping segment due to embedding error: {e}", flush=True)
                continue

    if len(embeddings) < 2:
        # Not enough speakers to cluster, just assign everyone to Speaker_1
        turns = [{"start": t["start"], "end": t["end"], "speaker": "Speaker_1"} for t in valid_turns]
    else:
        # 4. Clustering (Force exactly 2 speakers)
        print(f"Clustering {len(embeddings)} segments into 2 speakers...", flush=True)
        clusterer = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
        labels = clusterer.fit_predict(embeddings)
        
        turns = []
        for i, label in enumerate(labels):
            turns.append({
                "start": valid_turns[i]["start"],
                "end": valid_turns[i]["end"],
                "speaker": f"Speaker_{label + 1}"
            })

    # 5. Merge and Transcribe
    turns = merge_adjacent_turns(turns, max_gap_seconds=max_gap_seconds)
    
    # Flush VRAM for Whisper
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Transcribing {len(turns)} diarized segments...", flush=True)
    segments = []
    for turn in turns:
        start_idx = int(turn["start"] * 16000)
        end_idx = int(turn["end"] * 16000)
        segment_audio = np.ascontiguousarray(audio[start_idx:end_idx], dtype=np.float32)
        
        text = engine.transcribe(segment_audio, language_code, task=task).strip()
        if text:
            segments.append({
                "speaker": turn["speaker"],
                "start": turn["start"],
                "end": turn["end"],
                "text": text
            })

    # 6. Generate Plots
    import matplotlib.pyplot as plt
    from pyannote.core import Annotation, Segment
    
    plot_path = None
    cluster_plot_path = None
    
    try:
        os.makedirs("plots", exist_ok=True)
        
        # Plot 1: Timeline
        annotation = Annotation()
        for turn in turns:
            annotation[Segment(turn["start"], turn["end"])] = turn["speaker"]
        
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        from pyannote.core.notebook import notebook
        notebook.plot_annotation(annotation, ax=ax1, time=True, legend=True)
        ax1.set_title("Speaker Diarization Timeline")
        plt.tight_layout()
        plot_path = os.path.join("plots", "diarization_timeline.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig1)
        
        # Plot 2: Speaker Clusters (PCA)
        if len(embeddings) >= 3: # Need at least 3 points for a convex hull in some cases, but 2 for PCA
            print("Generating speaker cluster visualization...", flush=True)
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(np.array(embeddings))
            
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            colors = ['#4C72B0', '#C44E52'] # Blue and Red/Pinkish
            
            for label_id in range(2):
                idx = np.where(labels == label_id)[0]
                if len(idx) == 0: continue
                
                points = embeddings_2d[idx]
                ax2.scatter(points[:, 0], points[:, 1], c=colors[label_id], 
                           label=f"Speaker {label_id + 1}", s=50, edgecolors='white', alpha=0.8)
                
                # Draw Convex Hull if there are enough points
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax2.plot(points[simplex, 0], points[simplex, 1], color=colors[label_id], linestyle='--', alpha=0.6)
                        
                        # Add label in the middle of the cluster
                        center = np.mean(points, axis=0)
                        ax2.text(center[0], center[1], f"SPEAKER {label_id + 1}", 
                                fontsize=12, fontweight='bold', ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                    except:
                        pass # Hull might fail for collinear points
            
            ax2.set_title("Speaker Diarization Clusters (PCA Visualization)")
            ax2.set_xlabel("PCA component 1")
            ax2.set_ylabel("PCA component 2")
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            cluster_plot_path = os.path.join("plots", "speaker_clusters.png")
            plt.savefig(cluster_plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig2)
            print(f"Cluster plot saved to {cluster_plot_path}", flush=True)

    except Exception as e:
        print(f"Failed to plot diarization: {e}", flush=True)

    return {
        "segments": segments,
        "transcript": format_diarized_transcript(segments),
        "plot_path": plot_path,
        "cluster_plot_path": cluster_plot_path
    }
