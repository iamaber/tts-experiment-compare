# FastPitch voice cloning with TitaNet

Multi-speaker TTS with a speaker conditioning layer.

## Quick start

### 1. Extract your voice embedding

```bash
uv run python speaker_encoder.py custom-voice/record.wav my_voice.pt titanet
```

### 2. Generate speech with your voice

```bash
uv run python custom_voice_tts.py my_voice.pt "Your text here" output.wav
```

## Project structure

### Core files
- `speaker_encoder.py`: speaker encoder with TitaNet, Resemblyzer, and ECAPA backends
- `speaker_adapter.py`: speaker adaptation layer and the hook that adds it to FastPitch's encoder output
- `speaker_conditioning.py`: FastPitch + HiFi-GAN pipeline that uses the speaker adapter
- `custom_voice_tts.py`: custom voice TTS generator
- `tts.py`: multi-speaker TTS (12,800 pre-trained voices)

### Data
- `custom-voice/record.wav`: your voice recording (155s)
- `custom-voice/record.txt`: transcript
- `my_voice_titanet.pt`: extracted voice embedding (192D)

### Experiments
- `experiments/`: earlier approaches

## Architecture

### Speaker conditioning

```
Text → FastPitch Encoder → [Batch, Seq, 384]
                                  ↓
Voice → TitaNet → [192D] → Projection → [384D] → Broadcast
                                                      ↓
                                                  ADD (Global Conditioning)
                                                      ↓
                                            Duration/Pitch Predictors
```

TitaNet extracts a 192D speaker embedding. A projection layer maps it to 384D, the FastPitch hidden dimension. A forward hook on the FastPitch encoder adds the result to every position of the encoder output (global conditioning), so the duration and pitch predictors see it.

## Speaker encoder backends

| Backend | Dimension | Quality | Speed | Use case |
|---------|-----------|---------|-------|----------|
| TitaNet | 192D | 5/5 | Fast | Recommended (NeMo) |
| Resemblyzer | 256D | 4/5 | Medium | Alternative |
| ECAPA | 192D | 4/5 | Fast | SpeechBrain users |

## Current status

### Working
- Speaker embedding extraction (TitaNet)
- Projection layer (192D to 384D)
- Global conditioning layer
- Architecture demo

### Limitations
- The speaker adapter is untrained, so output does not match the reference voice yet.

### For real voice cloning
- Fine-tune FastPitch on your voice data (see `voice-swap-NeMo`), or
- Train the speaker adapter on multi-speaker data.

## System requirements

- Python 3.10+
- NVIDIA GPU (CUDA)
- NeMo 2.6.0+
- PyTorch with CUDA support

## Documentation

- `experiments/README.md`: approaches tried

## Why TitaNet

TitaNet is part of NVIDIA NeMo, the same toolkit as FastPitch. It is a speaker verification model that produces 192D embeddings and runs on the GPU.

## Next steps

1. Use existing voices. Try speakers 0-12799 in `tts.py`.
2. Fine-tune FastPitch on your voice.
3. Train the speaker adapter.
