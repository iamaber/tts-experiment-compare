import torch
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.asr.models import EncDecSpeakerLabelModel

from speaker_adapter import SpeakerAdaptationLayer, speaker_conditioned


class FastPitchWithSpeakerConditioning:
    def __init__(self, verbose=True):
        self.verbose = verbose

        if verbose:
            print("Loading models...")

        self.fastpitch = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
        self.fastpitch.eval()

        self.hifigan = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
        self.hifigan.eval()

        self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.speaker_encoder.eval()

        self.hidden_dim = self.fastpitch.fastpitch.encoder.d_model
        speaker_dim = self.speaker_encoder.decoder.final.in_features
        self.speaker_adapter = SpeakerAdaptationLayer(
            input_embedding_dim=speaker_dim, model_hidden_dim=self.hidden_dim
        )

        device = next(self.fastpitch.parameters()).device
        self.speaker_adapter = self.speaker_adapter.to(device)

        if verbose:
            print("✅ Models loaded")
            print(f"   FastPitch: Single-speaker ({self.hidden_dim}D hidden)")
            print(f"   TitaNet: Speaker encoder ({speaker_dim}D)")
            print(
                f"   Adapter: {speaker_dim}D → {self.hidden_dim}D projection + global conditioning"
            )

    def extract_speaker_embedding(self, audio_path, target_sr=16000):
        if self.verbose:
            print(f"Extracting speaker embedding: {audio_path}")

        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform).float()

        if sr != target_sr:
            import torchaudio.transforms as T

            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        audio_length = torch.tensor([waveform.shape[1]])

        device = next(self.speaker_encoder.parameters()).device
        waveform = waveform.to(device)
        audio_length = audio_length.to(device)

        with torch.no_grad():
            _, embeddings = self.speaker_encoder(
                input_signal=waveform, input_signal_length=audio_length
            )
            embedding = embeddings.squeeze()

        if self.verbose:
            print(f"✅ Extracted: {embedding.shape}")

        return embedding

    def generate_with_speaker(
        self, text, speaker_embedding, pace=1.0, output="output.wav"
    ):
        if self.verbose:
            print(f"Generating speech with speaker conditioning...")

        tokens = self.fastpitch.parse(text)

        with (
            torch.no_grad(),
            speaker_conditioned(
                self.fastpitch, self.speaker_adapter, speaker_embedding
            ),
        ):
            spec = self.fastpitch.generate_spectrogram(tokens=tokens, pace=pace)

        audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().detach().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        sample_rate = self.hifigan.sample_rate
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, sample_rate)

        if self.verbose:
            print(f"✅ Saved: {output} ({len(audio) / sample_rate:.2f}s)")

        return output

    def demonstrate_architecture(self, speaker_embedding):
        print("\n📐 Demonstrating Speaker Conditioning Architecture:")

        batch_size = 1
        seq_len = 50
        hidden_dim = self.hidden_dim

        device = next(self.speaker_adapter.parameters()).device

        text_hidden = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        device = next(self.speaker_adapter.parameters()).device
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
        speaker_embedding = speaker_embedding.to(device)

        conditioned = self.speaker_adapter(text_hidden, speaker_embedding)

        print(f"   Text hidden states: {text_hidden.shape}")
        print(f"   Speaker embedding: {speaker_embedding.shape}")
        print(f"   Conditioned output: {conditioned.shape}")
        print(f"   ✅ Speaker identity added to every phoneme!")


if __name__ == "__main__":
    import sys

    tts = FastPitchWithSpeakerConditioning()

    if len(sys.argv) < 2:
        print("Usage: python speaker_conditioning.py <voice_audio.wav> [text]")
        print("\nThis demonstrates the CORRECT architecture for speaker conditioning.")
        sys.exit(1)

    voice_file = sys.argv[1]
    text = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "This demonstrates proper speaker conditioning."
    )

    embedding = tts.extract_speaker_embedding(voice_file)

    tts.demonstrate_architecture(embedding)

    tts.generate_with_speaker(text, embedding, output="output_conditioned.wav")

    print("\n⚠️  IMPORTANT:")
    print("   The speaker adapter is untrained, so the output won't match the voice")
    print("   until the adapter is trained on speaker data")
