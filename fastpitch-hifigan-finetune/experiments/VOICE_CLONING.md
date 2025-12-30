# Custom Voice Cloning Guide

## Quick Start

### Step 1: Extract Your Voice Embedding

```bash
# Record 10-30 seconds of your voice and save as WAV
# Then extract the embedding:
uv run python clone_voice.py my_voice.wav
```

This creates `custom_voice.pt` with your voice embedding.

### Step 2: Generate Speech with Your Voice

```bash
# Use your voice to generate speech:
uv run python custom_tts.py custom_voice.pt "Hello, this is my cloned voice!"
```

## Detailed Usage

### Recording Your Voice

**Requirements:**
- **Duration**: 10-30 seconds (longer is better)
- **Quality**: Clear, no background noise
- **Format**: WAV, MP3, or any audio format
- **Content**: Read naturally, vary intonation

**Tips:**
- Speak clearly and naturally
- Include different emotions/tones
- Avoid background noise
- Use good microphone quality

### Extract Embedding

```python
from clone_voice import clone_voice

# Extract from your audio
clone_voice("my_voice.wav", output_embedding="my_voice.pt")
```

### Use Custom Voice

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

### Advanced: Multiple Custom Voices

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

## How It Works

1. **Speaker Encoder**: Uses SpeechBrain's ECAPA-TDNN model
2. **Embedding Extraction**: Converts your voice to a 192D vector
3. **Embedding Injection**: Replaces a speaker's embedding in FastPitch
4. **Synthesis**: FastPitch uses your embedding to generate speech

## Troubleshooting

### "Dimension mismatch" Warning

The script automatically adapts embeddings to the correct size. This is normal and usually works fine.

### Poor Quality Output

- **Solution 1**: Use longer/better quality reference audio
- **Solution 2**: Try different speakers to replace (0-12799)
- **Solution 3**: Adjust pace parameter

### Voice Doesn't Sound Like You

- Record more natural speech (10-30 seconds)
- Ensure good audio quality
- Try multiple recordings and pick the best

## File Structure

```
my_voice.wav          # Your voice recording
custom_voice.pt       # Extracted embedding
custom_output.wav     # Generated speech
```

## Complete Example

```bash
# 1. Record your voice (save as my_voice.wav)

# 2. Extract embedding
uv run python clone_voice.py my_voice.wav

# 3. Generate speech
uv run python custom_tts.py custom_voice.pt "Testing my cloned voice"

# 4. Listen to custom_output.wav
```

## Notes

- First run downloads speaker encoder (~80MB)
- Embedding extraction takes ~5 seconds
- Works with any audio format (WAV, MP3, etc.)
- Can replace any of the 12,800 speakers
