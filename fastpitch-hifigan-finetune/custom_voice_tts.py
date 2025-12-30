"""
Custom Voice TTS - Generate speech with your voice
Uses speaker_encoder.py + speaker_conditioning.py architecture
"""

import torch
import sys
from pathlib import Path
from speaker_encoder import SpeakerEncoder
from speaker_conditioning import FastPitchWithSpeakerConditioning


def generate_custom_voice(
    text: str,
    voice_audio: str = None,
    voice_embedding: str = None,
    output: str = "custom_voice_output.wav",
    pace: float = 1.0
):
    """
    Generate speech with custom voice.
    
    Args:
        text: Text to synthesize
        voice_audio: Path to voice audio (will extract embedding)
        voice_embedding: Path to pre-extracted embedding (.pt file)
        output: Output audio file
        pace: Speaking speed
    """
    # Initialize TTS
    print("Initializing TTS system...")
    tts = FastPitchWithSpeakerConditioning(verbose=False)
    
    # Get speaker embedding
    if voice_embedding and Path(voice_embedding).exists():
        # Load pre-extracted embedding
        print(f"Loading embedding: {voice_embedding}")
        embedding = torch.load(voice_embedding)
        print(f"✅ Loaded: {embedding.shape}")
    elif voice_audio and Path(voice_audio).exists():
        # Extract embedding from audio
        print(f"Extracting embedding from: {voice_audio}")
        encoder = SpeakerEncoder(backend="titanet")
        embedding = encoder.extract(voice_audio)
        print(f"✅ Extracted: {embedding.shape}")
        
        # Save for future use
        embedding_file = "extracted_embedding.pt"
        encoder.save_embedding(embedding, embedding_file)
    else:
        raise ValueError("Must provide either voice_audio or voice_embedding")
    
    # Demonstrate architecture
    print("\n📐 Speaker Conditioning Architecture:")
    tts.demonstrate_architecture(embedding)
    
    # Generate speech
    print(f"\n🎤 Generating: '{text}'")
    tts.generate_with_speaker(text, embedding, pace=pace, output=output)
    
    print(f"\n✅ Complete!")
    print(f"   Output: {output}")
    print(f"\n⚠️  Note: This uses the correct architecture (projection + conditioning)")
    print(f"   but doesn't modify FastPitch's forward pass yet.")
    print(f"   The output uses FastPitch's default voice, not your custom voice.")
    print(f"\n💡 For true voice cloning: Fine-tune FastPitch on your voice data")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python custom_voice_tts.py <voice_audio_or_embedding> <text> [output.wav] [pace]")
        print("\nExamples:")
        print("  # Using audio file")
        print("  python custom_voice_tts.py my_voice.wav \"Hello world\" output.wav 1.0")
        print("\n  # Using pre-extracted embedding")
        print("  python custom_voice_tts.py my_voice_titanet.pt \"Hello world\" output.wav 1.0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    text = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "custom_voice_output.wav"
    pace = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    # Determine if input is audio or embedding
    if input_file.endswith('.pt'):
        generate_custom_voice(text, voice_embedding=input_file, output=output, pace=pace)
    else:
        generate_custom_voice(text, voice_audio=input_file, output=output, pace=pace)
