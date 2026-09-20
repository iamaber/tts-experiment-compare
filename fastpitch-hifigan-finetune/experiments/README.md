# Experiments: voice cloning attempts

This folder holds the voice cloning approaches we tried with FastPitch.

## Files

### Experimental scripts
- `clone_voice.py`: Resemblyzer encoder (256D, zero-padding)
- `clone_voice_improved.py`: neural projection (untrained)
- `clone_voice_nemo.py`: TitaNet encoder (embedding replacement)
- `custom_tts.py`: TTS with embedding replacement
- `proper_voice_cloning.py`: single-speaker FastPitch attempt
- `fastpitch_projection.py`: projection layer (embedding replacement)

### Documentation
- `VOICE_CLONING.md`: initial voice cloning guide
- `VOICE_CLONING_COMPARISON.md`: comparison of all methods
- `QUALITY_ISSUES.md`: FastPitch quality analysis

### Embeddings
- `custom_voice.pt`: Resemblyzer (256D)
- `custom_voice_384d.pt`: neural projection (384D)
- `custom_voice_nemo.pt`: TitaNet (192D)
- `speaker_embedding.pt`: various experiments

## Why these are experimental

All these approaches replace speaker embeddings, which fails for four reasons:
- FastPitch embeddings are learned parameters.
- External encoders use different embedding spaces.
- Dimension mismatches require padding or projection.
- The output does not match the target voice.

## The approach in the root directory

- `speaker_encoder.py`: speaker encoder
- `speaker_conditioning.py`: global conditioning layer
- `custom_voice_tts.py`: TTS script

It projects the embedding and adds it to the encoder output (global conditioning) instead of replacing an embedding.
