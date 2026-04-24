# Whisper Fine-Tuning for Indic Languages (Hindi & Marathi)

A research-grade pipeline for fine-tuning OpenAI's Whisper model on Hindi and Marathi speech data using QLoRA (4-bit quantization). Designed for medical transcription research and thesis validation.

## 🚀 Features
- **QLoRA (4-bit) Fine-tuning:** Train Large/Medium models on consumer GPUs (e.g., RTX 4050 6GB).
- **Advanced Pre-processing:** Integrated **Silero VAD** for silence stripping and **Spectral Denoising** for audio cleaning.
- **IndicNLP Post-processing:** Orthographic normalization for Devanagari script to improve Character Error Rate (CER).
- **Research Dashboard:** A premium Gradio Web UI for comparing Base vs. Fine-tuned models with visual diff analysis.
- **Automated Plotting:** Generate Loss and WER curves automatically from training logs.

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Whisper-Fine-tuning
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   # Install Torch with CUDA 12.1 support
   pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
   
   # Install the rest
   pip install -r requirements.txt
   ```

## 📈 Usage

### 1. Data Preparation
Collect 10 hours of Hindi and Marathi data from Google FLEURS:
```bash
python prepare_dataset.py
```

### 2. Training (QLoRA)
To train on your local machine or a lab PC:
```bash
python train_whisper_qlora.py --model-name openai/whisper-medium --max-steps 1000
```

### 3. Evaluation & Plotting
Generate research graphs for your thesis:
```bash
python plot_training_logs.py --log-files output/whisper-small-hi-qlora/training_logs.csv --labels Hindi
```

### 4. Interactive Demo
Launch the premium web interface:
```bash
python app.py
```

## 📝 License
MIT
