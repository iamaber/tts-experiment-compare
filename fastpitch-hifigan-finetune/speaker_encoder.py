import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Union


class SpeakerEncoder:
    def __init__(self, backend="titanet"):
        """
        Args:
            backend: Encoder type - "titanet", "resemblyzer", or "ecapa"
        """
        self.backend = backend
        self.encoder = None
        self.embedding_dim = None

        self._load_encoder()

    def _load_encoder(self):
        print(f"Loading {self.backend} speaker encoder...")

        if self.backend == "titanet":
            self._load_titanet()
        elif self.backend == "resemblyzer":
            self._load_resemblyzer()
        elif self.backend == "ecapa":
            self._load_ecapa()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print(f"✅ {self.backend} loaded - {self.embedding_dim}D embeddings")

    def _load_titanet(self):
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.encoder = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.encoder.eval()
        self.embedding_dim = self.encoder.decoder.final.in_features

    def _load_resemblyzer(self):
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder()
        self.embedding_dim = 256

    def _load_ecapa(self):
        from speechbrain.pretrained import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        self.embedding_dim = 192

    def extract(
        self, audio_path: Union[str, Path], target_sr: int = 16000
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio file.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (16000 for most encoders)

        Returns:
            Speaker embedding tensor
        """
        audio_path = str(audio_path)

        if self.backend == "titanet":
            return self._extract_titanet(audio_path, target_sr)
        elif self.backend == "resemblyzer":
            return self._extract_resemblyzer(audio_path)
        elif self.backend == "ecapa":
            return self._extract_ecapa(audio_path, target_sr)

    def _extract_titanet(self, audio_path: str, target_sr: int) -> torch.Tensor:
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform).float()

        if sr != target_sr:
            import torchaudio.transforms as T

            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        audio_length = torch.tensor([waveform.shape[1]])

        device = next(self.encoder.parameters()).device
        waveform = waveform.to(device)
        audio_length = audio_length.to(device)

        with torch.no_grad():
            _, embeddings = self.encoder(
                input_signal=waveform, input_signal_length=audio_length
            )
            embedding = embeddings.squeeze().cpu()

        return embedding

    def _extract_resemblyzer(self, audio_path: str) -> torch.Tensor:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(audio_path)

        embedding = self.encoder.embed_utterance(wav)
        embedding = torch.from_numpy(embedding).float()

        return embedding

    def _extract_ecapa(self, audio_path: str, target_sr: int) -> torch.Tensor:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        with torch.no_grad():
            embedding = self.encoder.encode_batch(waveform)
            embedding = embedding.squeeze()

        return embedding

    def save_embedding(self, embedding: torch.Tensor, output_path: Union[str, Path]):
        torch.save(embedding, output_path)
        print(f"💾 Saved: {output_path}")

    def load_embedding(self, embedding_path: Union[str, Path]) -> torch.Tensor:
        embedding = torch.load(embedding_path)
        print(f"📥 Loaded: {embedding_path}")
        return embedding


def extract_speaker_embedding(
    audio_path: str, output_path: str = "speaker_embedding.pt", backend: str = "titanet"
) -> torch.Tensor:
    """
    Convenience function to extract and save speaker embedding.

    Args:
        audio_path: Path to audio file
        output_path: Where to save embedding
        backend: Encoder type ("titanet", "resemblyzer", "ecapa")

    Returns:
        Speaker embedding tensor
    """
    encoder = SpeakerEncoder(backend=backend)

    print(f"Extracting from: {audio_path}")
    embedding = encoder.extract(audio_path)

    print(f"✅ Extracted: {embedding.shape}")
    encoder.save_embedding(embedding, output_path)

    return embedding


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python speaker_encoder.py <audio.wav> [output.pt] [backend]")
        print("\nBackends:")
        print("  titanet    - NVIDIA TitaNet (192D) [default]")
        print("  resemblyzer - Resemblyzer (256D)")
        print("  ecapa      - SpeechBrain ECAPA (192D)")
        print("\nExample:")
        print("  python speaker_encoder.py voice.wav speaker.pt titanet")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "speaker_embedding.pt"
    backend = sys.argv[3] if len(sys.argv) > 3 else "titanet"

    if not Path(audio_file).exists():
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)

    embedding = extract_speaker_embedding(audio_file, output_file, backend)

    print(f"\n✅ Done!")
    print(f"   Backend: {backend}")
    print(f"   Embedding: {embedding.shape}")
    print(f"   Saved to: {output_file}")
