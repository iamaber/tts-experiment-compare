import torch
import torch.nn as nn
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

from speaker_adapter import SpeakerAdaptationLayer, speaker_conditioned


class CustomFastPitch(nn.Module):
    def __init__(self, speaker_dim, pretrained_model_name="nvidia/tts_en_fastpitch"):
        super().__init__()

        print(f"Loading FastPitch: {pretrained_model_name}")
        self.fastpitch = FastPitchModel.from_pretrained(pretrained_model_name)
        self.fastpitch.eval()

        hidden_dim = self.fastpitch.fastpitch.encoder.d_model
        self.speaker_adapter = SpeakerAdaptationLayer(
            input_embedding_dim=speaker_dim, model_hidden_dim=hidden_dim
        )

        device = next(self.fastpitch.parameters()).device
        self.speaker_adapter = self.speaker_adapter.to(device)

        print(f"✅ Custom FastPitch ready")
        print(f"   Speaker adapter: {speaker_dim}D → {hidden_dim}D")

    def forward_with_speaker(self, text, speaker_embedding, pace=1.0):
        tokens = self.fastpitch.parse(text)

        with speaker_conditioned(
            self.fastpitch, self.speaker_adapter, speaker_embedding
        ):
            return self.fastpitch.generate_spectrogram(tokens=tokens, pace=pace)


class SpeakerEmbeddingEncoder:
    def __init__(self):
        print("Loading speaker encoder (TitaNet)...")
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.encoder = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.encoder.eval()
        self.embedding_dim = self.encoder.decoder.final.in_features

        print(f"✅ Speaker encoder loaded ({self.embedding_dim}D)")

    def extract(self, audio_path, target_sr=16000):
        import librosa

        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        waveform = torch.from_numpy(waveform).float()

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


class CustomVoiceTTS:
    def __init__(self, verbose=True):
        self.verbose = verbose

        if verbose:
            print("Initializing Custom Voice TTS...")

        self.speaker_encoder = SpeakerEmbeddingEncoder()

        self.fastpitch = CustomFastPitch(
            speaker_dim=self.speaker_encoder.embedding_dim,
            pretrained_model_name="nvidia/tts_en_fastpitch",
        )

        if verbose:
            print("Loading HiFiGAN vocoder...")
        self.hifigan = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
        self.hifigan.eval()
        if verbose:
            print("✅ All models ready")

    def extract_voice(self, audio_path):
        if self.verbose:
            print(f"Extracting speaker embedding from: {audio_path}")

        embedding = self.speaker_encoder.extract(audio_path)

        if self.verbose:
            print(f"✅ Extracted: {embedding.shape}")

        return embedding

    def generate(self, text, speaker_embedding, pace=1.0, output="output.wav"):
        if self.verbose:
            print(f"Generating: '{text}'")
            print(f"   Speaker conditioning: ACTIVE")
            print(f"   Pace: {pace}x")

        spec = self.fastpitch.forward_with_speaker(text, speaker_embedding, pace=pace)

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python custom_fastpitch.py <voice_audio.wav> <text> [output.wav] [pace]"
        )
        print("\nThis modifies FastPitch forward pass to inject speaker conditioning")
        print("into duration and pitch predictors for true voice cloning.")
        sys.exit(1)

    voice_file = sys.argv[1]
    text = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "custom_output.wav"
    pace = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    tts = CustomVoiceTTS()

    embedding = tts.extract_voice(voice_file)

    torch.save(embedding, "speaker_embedding.pt")
    print(f"💾 Saved speaker embedding to: speaker_embedding.pt")

    tts.generate(text, embedding, pace=pace, output=output)

    print("\n✅ Done")
    print("   The speaker adapter is untrained, so the output won't match the voice")
    print("   until the adapter is trained on speaker data")
