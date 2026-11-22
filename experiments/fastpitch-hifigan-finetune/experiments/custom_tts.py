import torch
from pathlib import Path
from tts import TTS


class CustomVoiceTTS(TTS):
    def __init__(self, custom_embedding_path=None, replace_speaker_id=0, **kwargs):
        """
        Args:
            custom_embedding_path: Path to your voice embedding (.pt file)
            replace_speaker_id: Which speaker ID to replace (default: 0)
            **kwargs: Other arguments for TTS
        """
        super().__init__(**kwargs)

        if custom_embedding_path:
            self.inject_custom_voice(custom_embedding_path, replace_speaker_id)

    def inject_custom_voice(self, embedding_path, speaker_id=0):
        """
        Replace a speaker embedding with your custom voice.

        Args:
            embedding_path: Path to embedding file
            speaker_id: Speaker ID to replace
        """
        print(f"Injecting custom voice into speaker {speaker_id}...")

        custom_embedding = torch.load(embedding_path)

        speaker_embeddings = self.fastpitch.fastpitch.speaker_emb.weight.data

        expected_dim = speaker_embeddings.shape[1]
        actual_dim = (
            custom_embedding.shape[0]
            if custom_embedding.ndim == 1
            else custom_embedding.shape[1]
        )

        if actual_dim != expected_dim:
            print(
                f"⚠️  Dimension mismatch: embedding is {actual_dim}D, need {expected_dim}D"
            )
            print(f"   Attempting to adapt...")

            if actual_dim > expected_dim:
                custom_embedding = custom_embedding[:expected_dim]
            else:
                padding = torch.zeros(expected_dim - actual_dim)
                custom_embedding = torch.cat([custom_embedding, padding])

        custom_embedding = custom_embedding / torch.norm(custom_embedding)

        speaker_embeddings[speaker_id] = custom_embedding.to(speaker_embeddings.device)

        print(f"✅ Custom voice injected into speaker {speaker_id}")
        print(f"   Now use speaker={speaker_id} to generate with your voice!")


def use_custom_voice(text, voice_embedding_path, output="output.wav", pace=1.0):
    """
    Generate speech with your custom voice.

    Args:
        text: Text to synthesize
        voice_embedding_path: Path to your voice embedding (.pt file)
        output: Output file path
        pace: Speaking pace

    Returns:
        Path to generated audio
    """
    tts = CustomVoiceTTS(
        custom_embedding_path=voice_embedding_path, replace_speaker_id=0
    )

    tts.generate(text, speaker=0, pace=pace, output=output)

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python custom_tts.py <embedding.pt> <text>")
        print("\nExample:")
        print('  python custom_tts.py custom_voice.pt "Hello, this is my voice!"')
        print("\nFirst create embedding with: python clone_voice.py your_audio.wav")
        sys.exit(1)

    embedding_file = sys.argv[1]
    text = sys.argv[2]

    if not Path(embedding_file).exists():
        print(f"❌ Error: Embedding file not found: {embedding_file}")
        print("   Run: python clone_voice.py your_audio.wav")
        sys.exit(1)

    print(f"Generating speech with custom voice...")
    output = use_custom_voice(text, embedding_file, output="custom_output.wav")
    print(f"\n✅ Done! Saved to: {output}")
