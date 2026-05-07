import gradio as gr
from dotenv import load_dotenv
load_dotenv()

import os
import time
import torch
from pathlib import Path
from asr_pipeline import get_engine
from speaker_diarization import format_diarized_transcript, transcribe_diarized_audio
# Removed difflib import

# Configuration
CHECKPOINTS = {
    "small": {"hi": "final_adapter", "mr": "final_adapter"},
    "medium": {"hi": "checkpoint-1500", "mr": "checkpoint-2000"}
}


# Removed generate_diff function

def process_audio(
    audio_path,
    model_size,
    language,
    model_select,
    task_type,
    apply_vad,
    apply_denoise,
    use_diarization,
):
    if audio_path is None:
        return "Please record audio.", 0.0, "", None, None

    engine = get_engine()
    start_time = time.time()

    # 1. Handle Model Switching
    model_name = f"openai/whisper-{model_size.lower()}"
    engine.switch_base_model(model_name)

    # 2. Handle Adapter Loading
    lang_code = "hi" if language == "Hindi" else "mr"
    task_str = "translate" if task_type == "Translate (English)" else "transcribe"
    use_lora = model_select and (task_str == "transcribe")
    
    if use_lora:
        ckpt = CHECKPOINTS[model_size.lower()][lang_code]
        adapter_path = Path("output") / f"whisper-{model_size.lower()}-{lang_code}-qlora" / ckpt
        try:
            engine.load_adapter(lang_code, str(adapter_path))
        except Exception as e:
            return f"Adapter not found at {adapter_path}", 0.0, "", f"Error: {str(e)}"
    else:
        engine.use_base_model()

    # 3. Process
    # Diarized transcripts need the original time axis, so VAD is disabled there.
    effective_vad = apply_vad and not use_diarization
    audio = engine.preprocess_audio(
        audio_path,
        apply_vad=effective_vad,
        apply_denoise=apply_denoise,
    )
    raw_text = engine.transcribe(audio, lang_code, task=task_str)

    diarized_text = ""
    diarized_plot = None
    diarized_cluster_plot = None
    if use_diarization:
        try:
            diarized_result = transcribe_diarized_audio(
                audio_path=audio_path,
                engine=engine,
                language_code=lang_code,
                hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                task=task_str,
                audio=audio,
                sample_rate=16000,
            )
            diarized_text = diarized_result["transcript"]
            diarized_plot = diarized_result.get("plot_path")
            diarized_cluster_plot = diarized_result.get("cluster_plot_path")
        except Exception as e:
            diarized_text = f"Speaker diarization failed: {str(e)}"

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    return raw_text, duration, diarized_text, diarized_plot, diarized_cluster_plot

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
.gradio-container { font-family: 'Inter', sans-serif !important; background: #0f172a !important; }
.glass-panel { background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(10px) !important; border-radius: 20px !important; padding: 20px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }
.main-header { text-align: center; background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; font-weight: 800 !important; }
.stat-card { background: #1e293b !important; border-radius: 15px !important; padding: 15px !important; text-align: center; }
#diff-box { background: #111827 !important; padding: 15px !important; border-radius: 10px !important; border-left: 4px solid #3b82f6 !important; }
"""

with gr.Blocks() as demo:
    with gr.Row(variant="compact"):
        gr.HTML("<h1 class='main-header' style='margin: 0; text-align: left; flex-grow: 8;'>INDIC-MED ASR DASHBOARD</h1>")
        gr.Image("vnit logo (1).jpg", show_label=False, container=False, width=100, height=100)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="glass-panel"):
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
            model_size = gr.Radio(["Small", "Medium"], label="Model Architecture", value="Medium")
            lang_select = gr.Radio(["Hindi", "Marathi"], label="Language", value="Hindi")
            task_type = gr.Radio(["Transcribe (Indic)", "Translate (English)"], label="Task", value="Transcribe (Indic)")
            model_select = gr.Checkbox(label="🚀 Use Fine-tuned Adapter (LoRA)", value=True)
            
            with gr.Accordion("Advanced Processing", open=False):
                vad_toggle = gr.Checkbox(label="VAD (Silence Stripping)", value=True)
                denoise_toggle = gr.Checkbox(label="Spectral Denoising", value=False)
                diarize_toggle = gr.Checkbox(label="Speaker Diarization", value=True)
            
            submit_btn = gr.Button("RUN ANALYSIS", variant="primary")
            
        with gr.Column(scale=2, elem_classes="glass-panel"):
            with gr.Row():
                output_time = gr.Number(label="Latency (s)")
                gr.Markdown("#### Status\n✅ Multi-Model Engine Ready")
            
            output_raw = gr.Textbox(label="Final Transcription", lines=8)
            output_diarized = gr.Textbox(label="Speaker-Diarized Transcript", lines=12)
            with gr.Row():
                output_plot = gr.Image(label="Diarization Timeline Plot", type="filepath")
                output_cluster_plot = gr.Image(label="Speaker Clusters (PCA Visualization)", type="filepath")

    submit_btn.click(
        fn=process_audio,
        inputs=[
            input_audio,
            model_size,
            lang_select,
            model_select,
            task_type,
            vad_toggle,
            denoise_toggle,
            diarize_toggle,
        ],
        outputs=[output_raw, output_time, output_diarized, output_plot, output_cluster_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True, css=custom_css)
