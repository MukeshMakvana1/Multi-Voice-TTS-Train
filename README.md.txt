# Kaliyo_Ai: Multi-Voice Hindi Text-to-Speech Training Studio

## Overview

**Kaliyo_Ai** एक ग्रेडियो-आधारित, मल्टी-वॉयस हिंदी टेक्स्ट टू स्पीच (TTS) मॉडल ट्रेनिंग प्लेटफॉर्म है, जिसे Windows 11 के लिए डिजाइन किया गया है और CPU-only setup पर भी शानदार चलता है। ElevenLabs-style अनुभव, आधुनिक यूआई, कस्टम Voice Training, आसान Dataset Upload के साथ।

---

## Features

- **Modern Gradio UI**
  - Tab-based navigation, Quick upload & training
  - Hindi-centric interface option
- **Multi-Voice Training Support**
  - किसी भी संख्या में स्पीकर या वॉयस (voice IDs)
- **Easy Dataset Upload**
  - `metadata.csv` (audio-file|text|speaker) + audio files (`.wav`, `.mp3`)
- **Advanced Training Controls**
  - Epochs, Batch Size, LR, Early Stopping, Mixed Precision, Gradient Accumulation
- **Text-to-Speech Generation**
  - Select Voice, Update pitch/speed, Generate & download synthesized speech
- **Live Training Logs**
  - Progress bar, status messages, error feedback
- **Model Export**
  - `.pth` format, ऑटोमेटेड Voice folder management
- **Professional Features Tab**
  - Data Augmentation, Monitoring, Multi-lingual settings, and 50+ advanced options
- **Simple code for easy customization and extension**
- **Hindi Dataset, Prompts, and Synthesis Support**
  - Optimized for Hindi input/output

---

## File Structure


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


---

## Dataset Preparation

### `metadata.csv` Format


udio1.wav|यह एक उदाहरण है।|speaker1
audio2.wav|आपका नाम क्या है?|speaker2
audio3.wav|मौका सभी को मिलता है।|speaker1

text

- **audio_filename:** datasets/wavs में मौजूद होना चाहिए
- **text:** Hindi transcription
- **speaker:** Unique identifier/voice name

---

## Installation

### 1. Clone Repository

git clone https://github.com/MukeshMakvana1/Kaliyo_Ai.git
cd Kaliyo_Ai

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

- **Minimum per speaker:** 1+ hour, 500+ clips per speaker recommended
- **Batch Size:** Start small (4~8 on CPU)
- **Epochs:** At least 50 for quick test, 100+ for quality
- **Audio Format:** Mono, 22050 Hz
- **Hindi Prompt:** Dataset, UI और output में पूरी तरह हिंदी सपोर्ट

---

## Advanced Features (Tab 5 Highlights)

- Data Augmentation: Time stretching, pitch shifting, background noise
- Architecture: Multi-head attention, normalization
- Optimization: AdamW, LR scheduler, gradient clipping
- Multi-speaker: Speaker embeddings, cloning, style transfer
- Monitoring: Loss curves, checkpointing, TensorBoard hooks
- Fine-tuning: LoRA, transfer learning support
- Multi-lingual: Phoneme conversion, accent modeling
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

- Raise issues, pull requests, ideas & improvements are welcome!
- Email: mukeshmakvana1@gmail.com

---

**सफल TTS मॉडलिंग और हिंदी वॉयस ट्रैनिंग के लिए शुभकामनाएँ! 🙌🚀**

requirements.txt
(आवश्यक Python लाइब्रेरी जहां त्वरित dependencies install हो जाएँ)

text
torch
torchaudio
gradio
numpy
pandas
soundfile
tqdm
metadata.csv (EXAMPLE in datasets/)

text
audio1.wav|यह एक उदाहरण है।|speaker1
audio2.wav|आपका नाम क्या है?|speaker2
audio3.wav|मौका सभी को मिलता है।|speaker1