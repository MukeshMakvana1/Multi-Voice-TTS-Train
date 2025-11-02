import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import pandas as pd
import numpy as np
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import soundfile as sf
from tqdm import tqdm
import random

# Create necessary folders
os.makedirs("voice", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/wavs", exist_ok=True)

# ==================== MODEL ARCHITECTURE (FIXED) ====================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        return self.transformer(x)

class MelDecoder(nn.Module):
    """FIXED: Prenet now accepts n_mels as input dimension"""
    def __init__(self, d_model=512, n_mels=80):
        super().__init__()
        # FIXED: Change input dimension from d_model to n_mels
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, 256),  # Changed from d_model to n_mels
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, d_model),  # Project to d_model
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, dim_feedforward=2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)
        
        self.mel_projection = nn.Linear(d_model, n_mels)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
        )
        
    def forward(self, memory, mel_input):
        # mel_input shape: (batch, seq_len, n_mels)
        x = self.prenet(mel_input)  # Now correctly handles n_mels input
        x = self.decoder(x, memory)
        mel_output = self.mel_projection(x)
        
        mel_postnet = self.postnet(mel_output.transpose(1, 2))
        mel_postnet = mel_postnet.transpose(1, 2)
        
        return mel_output, mel_postnet

class VoiceIDEncoder(nn.Module):
    def __init__(self, num_voices, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(num_voices, d_model)
        
    def forward(self, voice_ids):
        return self.embedding(voice_ids)

class TTSModel(nn.Module):
    def __init__(self, vocab_size, num_voices=100, d_model=512, n_mels=80):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model)
        self.voice_encoder = VoiceIDEncoder(num_voices, d_model)
        self.mel_decoder = MelDecoder(d_model, n_mels)
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.n_mels = n_mels
        
    def forward(self, text, voice_ids, mel_target=None, max_decoder_steps=1000):
        text_encoded = self.text_encoder(text)
        voice_emb = self.voice_encoder(voice_ids).unsqueeze(1)
        
        memory = text_encoded + voice_emb
        
        if mel_target is not None:
            # Training mode: use teacher forcing
            # Shift mel_target by 1 frame (add zero frame at start)
            batch_size = mel_target.size(0)
            zero_frame = torch.zeros(batch_size, 1, self.n_mels).to(mel_target.device)
            mel_input = torch.cat([zero_frame, mel_target[:, :-1, :]], dim=1)
        else:
            # Inference mode: start with zero frame
            batch_size = text.size(0)
            mel_input = torch.zeros(batch_size, 1, self.n_mels).to(text.device)
        
        durations = self.duration_predictor(text_encoded).squeeze(-1)
        mel_output, mel_postnet = self.mel_decoder(memory, mel_input)
        
        return mel_output, mel_postnet, durations

# ==================== VOCODER (SIMPLIFIED) ====================
class Vocoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.conv_pre = nn.Conv1d(n_mels, 512, 7, 1, padding=3)
        
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
        ])
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(256, 256, 3, 1, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 3, 1, padding=1),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 3, 1, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, 128, 3, 1, padding=1),
            ),
            nn.Sequential(
                nn.Conv1d(64, 64, 3, 1, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, 3, 1, padding=1),
            ),
            nn.Sequential(
                nn.Conv1d(32, 32, 3, 1, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(32, 32, 3, 1, padding=1),
            ),
        ])
        
        self.conv_post = nn.Conv1d(32, 1, 7, 1, padding=3)
        
    def forward(self, mel):
        x = self.conv_pre(mel)
        
        for up, res in zip(self.ups, self.res_blocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = res(x) + x
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x.squeeze(1)

# ==================== DATASET ====================
class TTSDataset(Dataset):
    def __init__(self, metadata_path, dataset_dir, max_text_len=200, max_mel_len=800):
        self.dataset_dir = Path(dataset_dir)
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        
        # Load metadata
        df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'text', 'speaker'])
        self.data = df.to_dict('records')
        
        # Build character vocabulary
        all_text = ' '.join(df['text'].tolist())
        self.chars = sorted(list(set(all_text)))
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<EOS>'] = len(self.chars) + 1
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Speaker to ID mapping
        speakers = df['speaker'].unique()
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(speakers)}
        self.num_speakers = len(speakers)
        
        print(f"✅ Dataset loaded: {len(self.data)} samples, {self.num_speakers} speakers")
        
    def __len__(self):
        return len(self.data)
    
    def text_to_sequence(self, text):
        return [self.char_to_idx.get(c, 0) for c in text] + [self.char_to_idx['<EOS>']]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio_path = self.dataset_dir / item['audio_file']
        if not audio_path.exists():
            audio_path = self.dataset_dir / 'wavs' / item['audio_file']
        
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample to 22050 if needed
            if sr != 22050:
                resampler = torchaudio.transforms.Resample(sr, 22050)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Convert to mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                f_min=0,
                f_max=8000
            )
            mel = mel_transform(waveform).squeeze(0).T
            
            # Log scale
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            # Text to sequence
            text_seq = self.text_to_sequence(item['text'])
            
            # Speaker ID
            speaker_id = self.speaker_to_id[item['speaker']]
            
            # Padding/Truncation
            if len(text_seq) > self.max_text_len:
                text_seq = text_seq[:self.max_text_len]
            else:
                text_seq = text_seq + [0] * (self.max_text_len - len(text_seq))
            
            if mel.size(0) > self.max_mel_len:
                mel = mel[:self.max_mel_len, :]
            else:
                padding = torch.zeros(self.max_mel_len - mel.size(0), 80)
                mel = torch.cat([mel, padding], dim=0)
            
            return {
                'text': torch.LongTensor(text_seq),
                'mel': mel,
                'speaker_id': speaker_id,
                'audio_path': str(audio_path)
            }
        except Exception as e:
            print(f"⚠️ Error loading {audio_path}: {e}")
            # Return dummy data
            return {
                'text': torch.zeros(self.max_text_len, dtype=torch.long),
                'mel': torch.zeros(self.max_mel_len, 80),
                'speaker_id': 0,
                'audio_path': ''
            }

# ==================== TRAINING ====================
class TTSTrainer:
    def __init__(self, model, vocoder, device='cpu'):
        self.model = model.to(device)
        self.vocoder = vocoder.to(device)
        self.device = device
        self.training_history = []
        
    def train(self, dataloader, epochs, learning_rate, save_dir, progress_callback=None):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        mel_loss_fn = nn.L1Loss()
        duration_loss_fn = nn.MSELoss()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(dataloader)
            
            for batch_idx, batch in enumerate(dataloader):
                text = batch['text'].to(self.device)
                mel = batch['mel'].to(self.device)
                speaker_id = torch.LongTensor(batch['speaker_id']).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    mel_output, mel_postnet, durations = self.model(text, speaker_id, mel)
                    
                    # Calculate losses
                    mel_loss = mel_loss_fn(mel_output, mel) + mel_loss_fn(mel_postnet, mel)
                    
                    # Duration target
                    duration_target = torch.ones_like(durations)
                    duration_loss = duration_loss_fn(durations, duration_target)
                    
                    loss = mel_loss + duration_loss * 0.1
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"⚠️ Error in batch {batch_idx}: {e}")
                    continue
                
                if progress_callback and batch_idx % 5 == 0:
                    progress = (epoch * num_batches + batch_idx) / (epochs * num_batches)
                    progress_callback(progress, f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_dir, "best_model.pth")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch + 1, avg_loss)
            
            if progress_callback:
                progress_callback((epoch + 1) / epochs, f"✅ Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        
        # Save final model
        self.save_model(save_dir, "final_model.pth")
        
        return self.training_history
    
    def save_checkpoint(self, save_dir, epoch, loss):
        checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'vocoder_state_dict': self.vocoder.state_dict(),
            'loss': loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_model(self, save_dir, filename):
        model_path = Path(save_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocoder_state_dict': self.vocoder.state_dict()
        }, model_path)
        print(f"💾 Model saved: {model_path}")

# ==================== INFERENCE ====================
def synthesize_speech(model, vocoder, text, speaker_id, char_to_idx, device='cpu'):
    model.eval()
    vocoder.eval()
    
    with torch.no_grad():
        # Convert text to sequence
        text_seq = [char_to_idx.get(c, 0) for c in text] + [char_to_idx.get('<EOS>', 1)]
        text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(device)
        speaker_tensor = torch.LongTensor([speaker_id]).to(device)
        
        # Generate mel spectrogram
        mel_output, mel_postnet, _ = model(text_tensor, speaker_tensor, mel_target=None)
        
        # Convert mel to audio
        audio = vocoder(mel_postnet.transpose(1, 2))
        
        return audio.cpu().numpy()

# ==================== GLOBAL VARIABLES ====================
global_model = None
global_vocoder = None
global_trainer = None
global_dataset = None
global_config = {}

# ==================== GRADIO FUNCTIONS ====================

def upload_dataset(metadata_file, audio_files):
    try:
        if metadata_file is None:
            return "❌ Please upload metadata.csv file"
        
        metadata_path = "datasets/metadata.csv"
        shutil.copy(metadata_file, metadata_path)
        
        wavs_dir = Path("datasets/wavs")
        wavs_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_count = 0
        if audio_files:
            for audio_file in audio_files:
                audio_name = Path(audio_file).name
                audio_path = wavs_dir / audio_name
                shutil.copy(audio_file, audio_path)
                uploaded_count += 1
        
        df = pd.read_csv(metadata_path, sep='|', header=None)
        num_samples = len(df)
        num_speakers = df[2].nunique() if len(df.columns) > 2 else 1
        
        return f"""✅ Dataset uploaded successfully!
📊 Total samples: {num_samples}
🎤 Number of speakers: {num_speakers}
📁 Metadata: {metadata_path}
🎵 Audio files uploaded: {uploaded_count}
📂 Audio directory: {wavs_dir}
"""
    except Exception as e:
        return f"❌ Error uploading dataset: {str(e)}"

def initialize_model(vocab_size, num_voices, hidden_dim, num_layers, learning_rate):
    global global_model, global_vocoder, global_trainer, global_config
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        global_model = TTSModel(vocab_size=vocab_size, num_voices=num_voices, d_model=hidden_dim)
        global_vocoder = Vocoder(n_mels=80)
        global_trainer = TTSTrainer(global_model, global_vocoder, device=device)
        
        global_config = {
            'vocab_size': vocab_size,
            'num_voices': num_voices,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'device': device
        }
        
        total_params = sum(p.numel() for p in global_model.parameters())
        vocoder_params = sum(p.numel() for p in global_vocoder.parameters())
        
        return f"""✅ Model initialized successfully!

🔧 Device: {device.upper()}
📊 TTS Model parameters: {total_params:,}
🎵 Vocoder parameters: {vocoder_params:,}
💾 Total parameters: {(total_params + vocoder_params):,}

⚙️ Configuration:
- Vocabulary size: {vocab_size}
- Number of voices: {num_voices}
- Hidden dimension: {hidden_dim}
- Transformer layers: {num_layers}
"""
    except Exception as e:
        return f"❌ Error initializing model: {str(e)}"

def start_training(epochs, batch_size, learning_rate, save_interval, use_mixed_precision, 
                  gradient_accumulation, warmup_steps, max_grad_norm, early_stopping_patience,
                  progress=gr.Progress()):
    global global_trainer, global_dataset, global_config, global_model, global_vocoder
    
    try:
        if global_trainer is None:
            return "❌ Please initialize the model first!"
        
        metadata_path = "datasets/metadata.csv"
        dataset_dir = "datasets"
        
        if not os.path.exists(metadata_path):
            return "❌ Please upload dataset first!"
        
        progress(0, desc="Loading dataset...")
        global_dataset = TTSDataset(metadata_path, dataset_dir)
        
        # Update model with correct vocab and speaker count
        device = global_config.get('device', 'cpu')
        vocab_size = len(global_dataset.char_to_idx)
        num_speakers = global_dataset.num_speakers
        
        progress(0.05, desc="Reinitializing model with correct dimensions...")
        
        # Reinitialize model
        global_model = TTSModel(vocab_size=vocab_size, num_voices=num_speakers, 
                               d_model=global_config.get('hidden_dim', 512))
        global_vocoder = Vocoder(n_mels=80)
        global_trainer = TTSTrainer(global_model, global_vocoder, device=device)
        
        dataloader = DataLoader(global_dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=0, drop_last=True)
        
        progress(0.1, desc="Starting training...")
        
        def progress_callback(prog, message):
            progress(prog, desc=message)
        
        # Start training
        history = global_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir="voice",
            progress_callback=progress_callback
        )
        
        # Save training history
        with open("voice/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save config
        config_to_save = {
            'vocab_size': vocab_size,
            'num_speakers': num_speakers,
            'char_to_idx': global_dataset.char_to_idx,
            'idx_to_char': global_dataset.idx_to_char,
            'speaker_to_id': global_dataset.speaker_to_id
        }
        
        with open("voice/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        return f"""✅ Training completed successfully!

📊 Training Statistics:
- Total epochs: {len(history)}
- Final loss: {history[-1]['loss']:.4f}
- Best loss: {min(h['loss'] for h in history):.4f}

💾 Saved Files:
- voice/final_model.pth
- voice/best_model.pth
- voice/config.json
- voice/training_history.json

🎉 Model ready for inference!
"""
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""❌ Training error: {str(e)}

📋 Details:
{error_details}
"""

def generate_speech(text, voice_selection, temperature, speed, pitch):
    global global_model, global_vocoder, global_dataset
    
    try:
        if global_model is None:
            return None, "❌ Please load a model first!"
        
        if not text:
            return None, "❌ Please enter text to synthesize!"
        
        if global_dataset is None:
            config_path = "voice/config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    char_to_idx = config['char_to_idx']
            else:
                return None, "❌ Dataset configuration not found!"
        else:
            char_to_idx = global_dataset.char_to_idx
        
        device = global_config.get('device', 'cpu')
        
        audio = synthesize_speech(
            global_model, 
            global_vocoder, 
            text, 
            voice_selection,
            char_to_idx,
            device=device
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"voice/generated_voice{voice_selection}_{timestamp}.wav"
        sf.write(output_path, audio[0], 22050)
        
        return output_path, f"""✅ Speech generated!

📝 Text: {text[:50]}...
🎤 Voice ID: {voice_selection}
💾 File: {output_path}
"""
    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

def load_pretrained_model(model_path):
    global global_model, global_vocoder, global_trainer, global_dataset, global_config
    
    try:
        if not os.path.exists(model_path):
            return "❌ Model file not found!"
        
        config_path = "voice/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                vocab_size = config['vocab_size']
                num_speakers = config['num_speakers']
        else:
            vocab_size = 1000
            num_speakers = 100
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=device)
        
        global_model = TTSModel(vocab_size=vocab_size, num_voices=num_speakers)
        global_vocoder = Vocoder()
        
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_vocoder.load_state_dict(checkpoint['vocoder_state_dict'])
        
        global_trainer = TTSTrainer(global_model, global_vocoder, device)
        
        global_config = {
            'device': device,
            'vocab_size': vocab_size,
            'num_voices': num_speakers
        }
        
        return f"""✅ Model loaded!

📁 Path: {model_path}
🔧 Device: {device}
📊 Vocab: {vocab_size} | Speakers: {num_speakers}
"""
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==================== GRADIO UI ====================
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="🎙️ TTS Studio") as app:
        
        gr.Markdown("# 🎙️ TTS Training Studio\n### Multi-Voice Text-to-Speech AI Platform")
        
        with gr.Tabs():
            with gr.Tab("📁 Dataset"):
                gr.Markdown("### Upload Training Data")
                
                with gr.Row():
                    metadata_upload = gr.File(label="📄 metadata.csv", file_types=['.csv'])
                    audio_uploads = gr.File(label="🎵 Audio Files", file_count="multiple", file_types=['.wav', '.mp3'])
                
                upload_btn = gr.Button("📤 Upload", variant="primary")
                upload_status = gr.Textbox(label="Status", lines=8)
                
                upload_btn.click(upload_dataset, [metadata_upload, audio_uploads], upload_status)
            
            with gr.Tab("⚙️ Config"):
                with gr.Row():
                    with gr.Column():
                        vocab_size = gr.Slider(100, 5000, 1000, step=100, label="📖 Vocab Size")
                        num_voices = gr.Slider(1, 500, 100, label="🎤 Max Voices")
                        hidden_dim = gr.Slider(128, 1024, 512, step=128, label="🧠 Hidden Dim")
                    with gr.Column():
                        num_layers = gr.Slider(2, 12, 6, label="🔀 Layers")
                        learning_rate_init = gr.Number(0.0001, label="📈 Learning Rate")
                
                init_btn = gr.Button("🚀 Initialize", variant="primary")
                init_status = gr.Textbox(label="Status", lines=12)
                
                init_btn.click(initialize_model, [vocab_size, num_voices, hidden_dim, num_layers, learning_rate_init], init_status)
            
            with gr.Tab("🎓 Training"):
                with gr.Row():
                    with gr.Column():
                        epochs = gr.Slider(1, 1000, 50, label="🔄 Epochs")
                        batch_size = gr.Slider(1, 32, 4, label="📦 Batch Size")
                        learning_rate = gr.Number(0.0001, label="📈 LR")
                        save_interval = gr.Slider(1, 100, 10, label="💾 Save Every")
                    with gr.Column():
                        use_mixed_precision = gr.Checkbox(False, label="⚡ Mixed Precision")
                        gradient_accumulation = gr.Slider(1, 16, 1, label="📊 Grad Accum")
                        warmup_steps = gr.Slider(0, 10000, 1000, step=100, label="🔥 Warmup")
                        max_grad_norm = gr.Number(1.0, label="✂️ Grad Clip")
                        early_stopping_patience = gr.Slider(0, 50, 10, label="⏹️ Early Stop")
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                training_status = gr.Textbox(label="Progress", lines=12)
                
                train_btn.click(start_training, [epochs, batch_size, learning_rate, save_interval, 
                               use_mixed_precision, gradient_accumulation, warmup_steps, 
                               max_grad_norm, early_stopping_patience], training_status)
            
            with gr.Tab("🎤 Generate"):
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox("voice/final_model.pth", label="📁 Model Path")
                        load_btn = gr.Button("📥 Load Model")
                        load_status = gr.Textbox(label="Status", lines=4)
                        
                        input_text = gr.Textbox(label="📝 Text", lines=4, placeholder="Enter text...")
                        voice_selection = gr.Slider(0, 99, 0, label="🎭 Voice ID")
                        
                        with gr.Row():
                            temperature = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="🌡️ Temp")
                            speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⚡ Speed")
                            pitch = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="🎵 Pitch")
                        
                        generate_btn = gr.Button("🎙️ Generate", variant="primary")
                    
                    with gr.Column():
                        output_audio = gr.Audio(label="🔊 Output", type="filepath")
                        generation_status = gr.Textbox(label="Status", lines=8)
                
                load_btn.click(load_pretrained_model, model_path, load_status)
                generate_btn.click(generate_speech, [input_text, voice_selection, temperature, speed, pitch], 
                                  [output_audio, generation_status])
        
        gr.Markdown("""
        ---
        ### 📚 Quick Guide:
        1. Upload dataset (metadata.csv + audio files)
        2. Initialize model with desired parameters
        3. Train for 50-100 epochs (start small)
        4. Load trained model and generate speech
        
        **Format:** `audio.wav|text transcript|speaker_name`
        """)
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
