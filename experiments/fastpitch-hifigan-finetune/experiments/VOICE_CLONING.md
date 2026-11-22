# Custom voice cloning guide

## Quick start

### Step 1: extract your voice embedding

```bash
# Record 10-30 seconds of your voice and save as WAV
# Then extract the embedding:
uv run python clone_voice.py my_voice.wav
```

This creates `custom_voice.pt` with your voice embedding.

### Step 2: generate speech with your voice

```bash
# Use your voice to generate speech:
uv run python custom_tts.py custom_voice.pt "Hello, this is my cloned voice!"
```

## Detailed usage

### Recording your voice

Record 10-30 seconds of clear speech with no background noise. Longer recordings work better. WAV, MP3, and other audio formats work. Read naturally, vary intonation, and include different tones. Use a good microphone.

### Extract embedding

```python
from clone_voice import clone_voice

# Extract from your audio
clone_voice("my_voice.wav", output_embedding="my_voice.pt")
```

### Use custom voice

```python
from custom_tts import use_custom_voice

# Generate with your voice
use_custom_voice(
    text="This will sound like me!",
    voice_embedding_path="my_voice.pt",
    output="my_speech.wav",
    pace=1.0
)
```

### Multiple custom voices

```python
from custom_tts import CustomVoiceTTS

# Initialize
tts = CustomVoiceTTS()

# Inject voice 1 into speaker 0
tts.inject_custom_voice("voice1.pt", speaker_id=0)

# Inject voice 2 into speaker 1
tts.inject_custom_voice("voice2.pt", speaker_id=1)

# Use different voices
tts.generate("Voice 1 speaking", speaker=0, output="v1.wav")
tts.generate("Voice 2 speaking", speaker=1, output="v2.wav")
```

## How it works

1. SpeechBrain's ECAPA-TDNN model encodes the speaker.
2. It converts your voice to a 192D vector.
3. The script replaces one speaker's embedding in FastPitch with that vector.
4. FastPitch generates speech from your embedding.

## Troubleshooting

### "Dimension mismatch" warning

The script resizes embeddings to the expected size. This warning is expected.

### Poor quality output

- Use longer or cleaner reference audio.
- Replace a different speaker (0-12799).
- Adjust the pace parameter.

### Voice doesn't sound like you

- Record 10-30 seconds of natural speech.
- Check the audio quality.
- Make several recordings and pick the best one.

## File structure

```
my_voice.wav          # Your voice recording
custom_voice.pt       # Extracted embedding
custom_output.wav     # Generated speech
```

## Complete example

```bash
# 1. Record your voice (save as my_voice.wav)

# 2. Extract embedding
uv run python clone_voice.py my_voice.wav

# 3. Generate speech
uv run python custom_tts.py custom_voice.pt "Testing my cloned voice"

# 4. Listen to custom_output.wav
```

## Notes

- The first run downloads the speaker encoder (~80MB).
- Embedding extraction takes ~5 seconds.
- Any audio format works (WAV, MP3, etc.).
- You can replace any of the 12,800 speakers.
