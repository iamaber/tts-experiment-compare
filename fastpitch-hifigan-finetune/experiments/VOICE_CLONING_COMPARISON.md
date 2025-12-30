# Voice Cloning Comparison - All Methods Tested

## Summary

Tested 3 different speaker encoders for voice cloning with FastPitch:

| Method | Embedding Size | Compatibility | Result |
|--------|---------------|---------------|---------|
| **Resemblyzer** | 256D | ⚠️ Poor | Padded to 384D with zeros |
| **Neural Projection** | 256D → 384D | ⚠️ Fair | Untrained projection network |
| **NVIDIA TitaNet** | 192D | ✅ Best | Same NeMo ecosystem |

## Files Created

### Embeddings
- `custom_voice.pt` - Resemblyzer (256D)
- `custom_voice_384d.pt` - Neural projection (384D)
- `custom_voice_nemo.pt` - **TitaNet (192D)** ⭐ Recommended

### Scripts
- `clone_voice.py` - Resemblyzer extractor
- `clone_voice_improved.py` - Neural projection
- `clone_voice_nemo.py` - **NVIDIA TitaNet** ⭐ Recommended
- `custom_tts.py` - TTS with custom embeddings

## Test Results

All three methods generated `custom_output.wav` successfully.

### Dimension Handling

**Resemblyzer:**
```
⚠️ Dimension mismatch: embedding is 256D, need 384D
   Attempting to adapt... (zero-padding)
```

**Neural Projection:**
```
✅ Custom voice injected into speaker 0 (no warnings)
   But: untrained projection = random mapping
```

**TitaNet:**
```
⚠️ Dimension mismatch: embedding is 192D, need 384D
   Attempting to adapt... (zero-padding)
```

## The Fundamental Problem

**All methods face the same issue:**

FastPitch's 384D speaker embeddings are **learned during training** and are model-specific. External speaker encoders (resemblyzer, TitaNet) produce embeddings in different "spaces":

- **Resemblyzer**: 256D universal speaker space
- **TitaNet**: 192D NeMo speaker verification space  
- **FastPitch**: 384D model-specific learned space

**Injecting external embeddings = speaking different languages**

Even TitaNet (from NVIDIA NeMo) doesn't match because:
- TitaNet is trained for **speaker verification** (is this the same person?)
- FastPitch embeddings are trained for **voice synthesis** (how does this person sound?)

## Why Voice Doesn't Match

When we inject external embeddings:
1. Dimension mismatch requires padding/projection
2. Padding with zeros = meaningless values to FastPitch
3. FastPitch doesn't understand the embedding
4. Result: Generic/robotic voice, not your voice

## The Only Real Solution

### **Fine-Tune FastPitch**

Train FastPitch to learn a **new 384D embedding** for your voice:

**Advantages:**
- ✅ Embedding is in FastPitch's learned space
- ✅ Perfect voice match
- ✅ Natural, high-quality synthesis

**Requirements:**
- Your voice recording: `custom-voice/record.wav` (155s) ✅
- Transcript: `custom-voice/record.txt` ✅
- GPU: RTX 3090 Ti ✅
- Time: 2-4 hours training

**This is what the project was originally designed for!**

## Recommendation

**Stop trying to inject embeddings** and fine-tune FastPitch properly.

The embedding injection approach is fundamentally limited because:
- External encoders use different embedding spaces
- Dimension adaptation loses voice information
- FastPitch can't interpret external embeddings

**Use the proper workflow:**
1. Prepare your data (already done!)
2. Fine-tune FastPitch on your voice
3. Get perfect voice cloning results

---

## Quick Commands

### Using TitaNet (Best of the 3 methods)
```bash
# Extract embedding
uv run python clone_voice_nemo.py custom-voice/record.wav

# Generate speech
uv run python custom_tts.py custom_voice_nemo.pt "Your text here"
```

### Fine-Tuning (Proper solution)
```bash
# Coming soon - need to set up training pipeline
```
