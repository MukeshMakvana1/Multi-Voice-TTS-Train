# Multi-Voice-TTS-Train
# Kaliyo_Ai: Multi-Voice Hindi Text-to-Speech Training Studio  ## Overview  **Kaliyo_Ai** is a Gradio-based, multi-voice Hindi Text-to-Speech (TTS) model training studio. It is designed for Windows 11, works great on CPU-only setups, and offers an ElevenLabs-style experience, modern UI, custom voice training, and easy dataset upload.
Kaliyo_Ai/
├── ui_tts_train.py # Main Gradio app code
├── README.md # This file
├── requirements.txt # Python dependencies
├── voice/ # Model exports & generated voices
│ ├── final_model.pth
│ ├── best_model.pth
│ ├── config.json
│ ├── training_history.json
│ └── generated_xxx.wav
├── datasets/
│ ├── metadata.csv # Dataset metadata
│ └── wavs/
│ └── *.wav # Audio training data
├── logs/
│ └── ... # Training logs
└── checkpoints/
└── ... # Model checkpoints

text

---

## Dataset Preparation

### `metadata.csv` Format

audio1.wav|This is an example.|speaker1
audio2.wav|What is your name?|speaker2
audio3.wav|Everyone gets a chance.|speaker1

text

- **audio_filename:** Must exist in `datasets/wavs`
- **text:** English or Hindi transcription/sentence
- **speaker:** Unique identifier/voice name

---

## Installation

### 1. Clone Repository

git clone https://github.com/MukeshMakvana1/Kaliyo_Ai.git
cd Kaliyo_Ai

text

### 2. Install Requirements

pip install -r requirements.txt

text

### 3. (Optional) Install ffmpeg

Linux:
sudo apt install ffmpeg

text
Windows:
Download from https://ffmpeg.org or use a package manager.

---

## Usage

1. **Start the Studio:**
python ui_tts_train.py

text

2. **Browse to:**
http://localhost:7860

text

3. **Steps:**
- Tab 1: Upload dataset (`metadata.csv` + audio)
- Tab 2: Configure model architecture
- Tab 3: Train your model (adjust settings)
- Tab 4: Generate Hindi speech, experiment with speakers
- Tab 5: Explore advanced features

---

## Training Recommendations

- **Minimum per speaker:** 1+ hour, at least 500 clips per speaker recommended
- **Batch size:** Start small (4-8 on CPU)
- **Epochs:** At least 50 for a quick test, 100+ for high quality
- **Audio format:** Mono, 22050 Hz sample rate
- **Hindi prompt:** Hindi supported in dataset, UI, and output

---

## Advanced Features (Tab 5 Highlights)

- Data augmentation: Time stretching, pitch shifting, background noise
- Architecture: Multi-head attention, normalization
- Optimization: AdamW, LR scheduler, gradient clipping
- Multi-speaker: Speaker embeddings, cloning, style transfer
- Monitoring: Loss curves, checkpointing, TensorBoard hooks
- Fine-tuning: LoRA, transfer learning support
- Multilingual: Phoneme conversion, accent modeling
- Performance: ONNX export, dynamic batching, caching

---

## Requirements

- **Python 3.8+**
- **PyTorch (latest)**
- **Gradio (UI)**
- **torchaudio**
- **numpy, pandas, soundfile**

Optional:
- **ffmpeg for audio manipulation**
- **TensorBoard for monitoring**
- **GPU for faster training** (CPU-only supported and optimized)

---

## Troubleshooting

- **Upload errors:** Check file format (audio + proper metadata.csv)
- **Shape errors:** Use latest version, proper MelDecoder input dimension
- **Audio:** Only mono audio recommended; stereo will be converted automatically

---

## Credits

- Built by Mukesh Makvana & Contributors
- Inspired by Tacotron2, NVIDIA TTS, ElevenLabs
- Powered by Gradio, PyTorch, torchaudio

---

## License

`Kaliyo_Ai` is released under the MIT License.

---

## Contact & Contribution

- Issues, pull requests, ideas & improvements are welcome!
- Email: mukeshmakvana1@gmail.com

---

**Happy TTS modeling & Hindi voice training! 🙌🚀**
requirements.txt

text
torch
torchaudio
gradio
numpy
pandas
soundfile
tqdm
metadata.csv (EXAMPLE for datasets/)

text
audio1.wav|This is an example.|speaker1
audio2.wav|What is your name?|speaker2
audio3.wav|Everyone gets a chance.|speaker1
