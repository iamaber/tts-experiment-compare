# FastPitch Voice Cloning with TitaNet

Multi-speaker TTS with proper speaker conditioning architecture.

## 🎯 Quick Start

### 1. Extract Your Voice Embedding

```bash
uv run python speaker_encoder.py custom-voice/record.wav my_voice.pt titanet
```

### 2. Generate Speech with Your Voice

```bash
uv run python custom_voice_tts.py my_voice.pt "Your text here" output.wav
```

## 📁 Project Structure

### **Core Files**
- `speaker_encoder.py` - Universal speaker encoder (TitaNet/Resemblyzer/ECAPA)
- `speaker_conditioning.py` - Speaker conditioning architecture
- `custom_voice_tts.py` - Custom voice TTS generator
- `tts.py` - Multi-speaker TTS (12,800 pre-trained voices)

### **Data**
- `custom-voice/record.wav` - Your voice recording (155s)
- `custom-voice/record.txt` - Transcript
- `my_voice_titanet.pt` - Extracted voice embedding (192D)

### **Experiments**
- `experiments/` - Previous experimental approaches

## 🏗️ Architecture

### **Proper Speaker Conditioning**

```
Text → FastPitch Encoder → [Batch, Seq, 384]
                                  ↓
Voice → TitaNet → [192D] → Projection → [384D] → Broadcast
                                                      ↓
                                                  ADD (Global Conditioning)
                                                      ↓
                                            Duration/Pitch Predictors
```

**Key Points:**
- ✅ TitaNet extracts 192D speaker embedding
- ✅ Projection layer maps to 384D (FastPitch hidden dim)
- ✅ Global conditioning (add to encoder output)
- ⚠️ Requires FastPitch forward pass modification for full implementation

## 🎤 Speaker Encoder Backends

| Backend | Dimension | Quality | Speed | Use Case |
|---------|-----------|---------|-------|----------|
| **TitaNet** | 192D | ⭐⭐⭐⭐⭐ | Fast | Recommended (NeMo ecosystem) |
| Resemblyzer | 256D | ⭐⭐⭐⭐ | Medium | Alternative |
| ECAPA | 192D | ⭐⭐⭐⭐ | Fast | SpeechBrain users |

## 📊 Current Status

### **What Works:**
- ✅ Speaker embedding extraction (TitaNet)
- ✅ Projection layer (192D → 384D)
- ✅ Global conditioning architecture
- ✅ Architecture demonstration

### **Limitation:**
- ⚠️ Doesn't modify FastPitch forward pass yet
- ⚠️ Output uses default voice (not custom voice)

### **For True Voice Cloning:**
- Fine-tune FastPitch on your voice data, OR
- Modify FastPitch forward pass to inject conditioned states

## 🔧 System Requirements

- Python 3.10+
- NVIDIA GPU (CUDA)
- NeMo 2.6.0+
- PyTorch with CUDA support

## 📚 Documentation

- `experiments/README.md` - Experimental approaches tried
- See artifacts in `.gemini/antigravity/brain/` for detailed walkthroughs

## 💡 Why TitaNet?

- **Same ecosystem** - NVIDIA NeMo
- **Optimized** - Designed for NeMo models
- **High quality** - State-of-the-art speaker verification
- **Compact** - 192D embeddings
- **Fast** - GPU accelerated

## 🚀 Next Steps

1. **Use existing voices** - Try speakers 0-12799 in `tts.py`
2. **Fine-tune** - Train FastPitch on your voice (proper solution)
3. **Modify FastPitch** - Inject conditioning into forward pass
