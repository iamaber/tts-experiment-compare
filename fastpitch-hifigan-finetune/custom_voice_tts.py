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
    pace: float = 1.0,
):
    print("Initializing TTS system...")
    tts = FastPitchWithSpeakerConditioning(verbose=False)

    if voice_embedding and Path(voice_embedding).exists():
        print(f"Loading embedding: {voice_embedding}")
        embedding = torch.load(voice_embedding)
        print(f"✅ Loaded: {embedding.shape}")
    elif voice_audio and Path(voice_audio).exists():
        print(f"Extracting embedding from: {voice_audio}")
        encoder = SpeakerEncoder(backend="titanet")
        embedding = encoder.extract(voice_audio)
        print(f"✅ Extracted: {embedding.shape}")

        embedding_file = "extracted_embedding.pt"
        encoder.save_embedding(embedding, embedding_file)
    else:
        raise ValueError("Must provide either voice_audio or voice_embedding")

    print("\n📐 Speaker Conditioning Architecture:")
    tts.demonstrate_architecture(embedding)

    print(f"\n🎤 Generating: '{text}'")
    tts.generate_with_speaker(text, embedding, pace=pace, output=output)

    print(f"\n✅ Complete!")
    print(f"   Output: {output}")
    print(f"\n⚠️  Note: the speaker adapter is untrained, so the output")
    print(f"   won't match your voice until the adapter is trained.")
    print(f"\n💡 For true voice cloning: Fine-tune FastPitch on your voice data")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python custom_voice_tts.py <voice_audio_or_embedding> <text> [output.wav] [pace]"
        )
        print("\nExamples:")
        print("  # Using audio file")
        print('  python custom_voice_tts.py my_voice.wav "Hello world" output.wav 1.0')
        print("\n  # Using pre-extracted embedding")
        print(
            '  python custom_voice_tts.py my_voice_titanet.pt "Hello world" output.wav 1.0'
        )
        sys.exit(1)

    input_file = sys.argv[1]
    text = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "custom_voice_output.wav"
    pace = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if input_file.endswith(".pt"):
        generate_custom_voice(
            text, voice_embedding=input_file, output=output, pace=pace
        )
    else:
        generate_custom_voice(text, voice_audio=input_file, output=output, pace=pace)
