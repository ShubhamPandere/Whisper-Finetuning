import gradio as gr
import os
import time
import torch
from pathlib import Path
from asr_pipeline import get_engine
import difflib

# Configuration
ADAPTER_PATHS = {
    "hi": "output/whisper-small-hi-qlora/final_adapter",
    "mr": "output/whisper-small-mr-qlora/final_adapter"
}

# Image path for the banner (using the one just generated)
BANNER_PATH = r"C:\Users\pande\.gemini\antigravity\brain\2c000774-6ce2-4a9a-83a7-4c3c0160e90f\medical_asr_banner_1777034640520.png"

def generate_diff(str1, str2):
    if str1 == str2:
        return "✨ No changes needed. Text is already normalized."
    
    result = ""
    codes = difflib.SequenceMatcher(None, str1, str2).get_opcodes()
    for tag, i1, i2, j1, j2 in codes:
        if tag == 'equal':
            result += str1[i1:i2]
        elif tag == 'replace':
            result += f"<span style='color: #ff4b4b; text-decoration: line-through;'>{str1[i1:i2]}</span>"
            result += f"<span style='color: #00ff88; font-weight: bold;'>{str2[j1:j2]}</span>"
        elif tag == 'delete':
            result += f"<span style='color: #ff4b4b; text-decoration: line-through;'>{str1[i1:i2]}</span>"
        elif tag == 'insert':
            result += f"<span style='color: #00ff88; font-weight: bold;'>{str2[j1:j2]}</span>"
    return result

def process_audio(audio_path, language, use_lora, apply_vad, apply_denoise, apply_post):
    if audio_path is None:
        return "Please record or upload an audio file.", 0.0, "", "No audio provided."

    engine = get_engine()
    start_time = time.time()

    # 1. Load Model/Adapter
    lang_code = "hi" if language == "Hindi" else "mr"
    if use_lora:
        adapter_path = ADAPTER_PATHS[lang_code]
        try:
            engine.load_adapter(lang_code, adapter_path)
        except Exception as e:
            return f"Error loading adapter: {str(e)}", 0.0, "", "Model Error"
    else:
        engine.use_base_model()

    # 2. Pre-process (VAD + Denoising)
    audio = engine.preprocess_audio(audio_path, apply_vad=apply_vad, apply_denoise=apply_denoise)

    # 3. Transcribe (Handles >30s chunking)
    raw_text = engine.transcribe(audio, lang_code)

    # 4. Post-process
    final_text = raw_text
    if apply_post:
        final_text = engine.postprocess_text(raw_text, lang_code)

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    # Generate Visual Diff for Post-processing
    diff_html = generate_diff(raw_text, final_text)

    return raw_text, duration, final_text, diff_html

# Custom CSS for Premium Look
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0f172a !important; /* Deep Navy Background */
}

.glass-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin-bottom: 0px !important;
}

.stat-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    text-align: center;
}

#diff-box {
    background: #111827 !important;
    padding: 15px !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
    border-left: 4px solid #3b82f6 !important;
}

.record-btn {
    background: linear-gradient(45deg, #ef4444 0%, #f87171 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}

.submit-btn {
    background: linear-gradient(45deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    # Header Section
    with gr.Row():
        if os.path.exists(BANNER_PATH):
            gr.Image(BANNER_PATH, show_label=False, elem_id="banner", container=False)
    
    gr.HTML("<h1 class='main-header'>INDIC-MED ASR ENGINE</h1>")
    gr.HTML("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: -10px;'>Next-Gen Clinical Transcription for Hindi & Marathi</p>")
    
    with gr.Row(equal_height=True):
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=1, elem_classes="glass-panel"):
            gr.Markdown("### 🎙️ Capturing System")
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Patient/Doctor Audio")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Engine Parameters")
                lang_select = gr.Radio(["Hindi", "Marathi"], label="Primary Language", value="Hindi")
                model_select = gr.Checkbox(label="🚀 Fine-tuned Adapter (LoRA)", value=True)
                
            with gr.Group():
                gr.Markdown("### 🛠️ Neural Processing")
                vad_toggle = gr.Checkbox(label="Enable VAD (Silence Stripping)", value=True)
                denoise_toggle = gr.Checkbox(label="Enable Spectral Denoising", value=False)
                post_toggle = gr.Checkbox(label="Enable Orthographic Normalization", value=True)
            
            submit_btn = gr.Button("RUN ANALYSIS", variant="primary", elem_classes="submit-btn")
            
        # RIGHT COLUMN: OUTPUTS
        with gr.Column(scale=2, elem_classes="glass-panel"):
            with gr.Row():
                with gr.Column(elem_classes="stat-card"):
                    output_time = gr.Number(label="Processing Latency (s)", precision=2)
                with gr.Column(elem_classes="stat-card"):
                    gr.Markdown("#### System Status\n✅ Long Audio Support Enabled")

            with gr.Group():
                gr.Markdown("### 🔍 RAW TRANSCRIPTION (Base)")
                output_raw = gr.Textbox(show_label=False, lines=4, placeholder="Waiting for processing...")
                
            with gr.Group():
                gr.Markdown("### ✨ CLEANED CLINICAL TEXT (Post-processed)")
                output_post = gr.Textbox(show_label=False, lines=4, interactive=False)
                
            with gr.Group():
                gr.Markdown("### 🧪 POST-PROCESSING PROOF (Diff Analysis)")
                output_diff = gr.HTML(elem_id="diff-box", value="Submit audio to see normalization proof.")

    # Footer Logic
    submit_btn.click(
        fn=process_audio,
        inputs=[input_audio, lang_select, model_select, vad_toggle, denoise_toggle, post_toggle],
        outputs=[output_raw, output_time, output_post, output_diff]
    )

    gr.Markdown("<p style='text-align: center; color: #475569; padding: 20px;'>Developed for Medical Research & Thesis Validation | 2026</p>")

if __name__ == "__main__":
    demo.launch(share=False)
