# Voice cloning comparison: all methods tested

## Summary

We tested 3 speaker encoders for voice cloning with FastPitch:

| Method | Embedding size | Compatibility | Result |
|--------|---------------|---------------|---------|
| Resemblyzer | 256D | Poor | Padded to 384D with zeros |
| Neural projection | 256D to 384D | Fair | Untrained projection network |
| NVIDIA TitaNet | 192D | Best | Same NeMo toolkit |

## Files created

### Embeddings
- `custom_voice.pt`: Resemblyzer (256D)
- `custom_voice_384d.pt`: neural projection (384D)
- `custom_voice_nemo.pt`: TitaNet (192D), recommended

### Scripts
- `clone_voice.py`: Resemblyzer extractor
- `clone_voice_improved.py`: neural projection
- `clone_voice_nemo.py`: NVIDIA TitaNet, recommended
- `custom_tts.py`: TTS with custom embeddings

## Test results

All three methods generated `custom_output.wav`.

### Dimension handling

Resemblyzer:
```
⚠️ Dimension mismatch: embedding is 256D, need 384D
   Attempting to adapt... (zero-padding)
```

Neural projection:
```
✅ Custom voice injected into speaker 0 (no warnings)
   But: untrained projection = random mapping
```

TitaNet:
```
⚠️ Dimension mismatch: embedding is 192D, need 384D
   Attempting to adapt... (zero-padding)
```

## The underlying problem

FastPitch learns its 384D speaker embeddings during training, and they are specific to the model. External speaker encoders (Resemblyzer, TitaNet) produce embeddings in different spaces:

- Resemblyzer: 256D general speaker space
- TitaNet: 192D NeMo speaker verification space
- FastPitch: 384D learned space specific to the model

FastPitch cannot interpret embeddings from another space.

Even TitaNet, also from NVIDIA NeMo, doesn't match:
- TitaNet is trained for speaker verification (is this the same person?).
- FastPitch embeddings are trained for synthesis (how does this person sound?).

## Why the voice doesn't match

When we inject external embeddings:
1. The dimension mismatch requires padding or projection.
2. Zero padding adds values that mean nothing to FastPitch.
3. FastPitch cannot interpret the embedding.
4. The output is a generic, robotic voice, not your voice.

## Fine-tune FastPitch

Fine-tuning trains FastPitch to learn a new 384D embedding for your voice. That embedding lives in FastPitch's learned space, so the model can use it for synthesis.

Requirements:
- Your voice recording: `custom-voice/record.wav` (155s), available
- Transcript: `custom-voice/record.txt`, available
- GPU: RTX 3090 Ti, available
- Time: 2-4 hours of training

The project was originally built for this workflow.

## Recommendation

Stop injecting embeddings and fine-tune FastPitch.

Embedding injection cannot work here because:
- External encoders use different embedding spaces.
- Dimension adaptation loses voice information.
- FastPitch can't interpret external embeddings.

Workflow:
1. Prepare your data (done).
2. Fine-tune FastPitch on your voice.
3. Generate speech with the fine-tuned model.

---

## Quick commands

### Using TitaNet (best of the 3 methods)
```bash
# Extract embedding
uv run python clone_voice_nemo.py custom-voice/record.wav

# Generate speech
uv run python custom_tts.py custom_voice_nemo.pt "Your text here"
```

### Fine-tuning
```bash
# Coming soon - need to set up training pipeline
```
