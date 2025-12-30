# Experiments - Voice Cloning Attempts

This folder contains experimental implementations and approaches we tried for voice cloning with FastPitch.

## Files

### **Experimental Scripts**
- `clone_voice.py` - Resemblyzer encoder (256D, zero-padding)
- `clone_voice_improved.py` - Neural projection (untrained)
- `clone_voice_nemo.py` - TitaNet encoder (embedding replacement)
- `custom_tts.py` - TTS with embedding replacement
- `proper_voice_cloning.py` - Single-speaker FastPitch attempt
- `fastpitch_projection.py` - Projection layer (embedding replacement)

### **Documentation**
- `VOICE_CLONING.md` - Initial voice cloning guide
- `VOICE_CLONING_COMPARISON.md` - Comparison of all methods
- `QUALITY_ISSUES.md` - FastPitch quality analysis

### **Embeddings**
- `custom_voice.pt` - Resemblyzer (256D)
- `custom_voice_384d.pt` - Neural projection (384D)
- `custom_voice_nemo.pt` - TitaNet (192D)
- `speaker_embedding.pt` - Various experiments

## Why These Are Experimental

All these approaches tried to **replace** speaker embeddings, which doesn't work because:
- FastPitch embeddings are learned parameters
- External encoders use different embedding spaces
- Dimension mismatches require padding/projection
- Results don't match the target voice

## The Correct Approach

See the root directory for the proper implementation:
- `speaker_encoder.py` - Universal speaker encoder
- `speaker_conditioning.py` - Proper global conditioning architecture
- `custom_voice_tts.py` - Final TTS implementation

The correct approach uses **projection + global conditioning** (adding to encoder output), not embedding replacement.
