import torch
import torch.nn as nn
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class SpeakerProjection(nn.Module):
    def __init__(self, input_dim=192, output_dim=384):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(self, speaker_embedding):
        return self.projection(speaker_embedding)


class FastPitchWithSpeakerProjection:
    def __init__(self, verbose=True):
        self.verbose = verbose

        if verbose:
            print("Loading models...")

        self.fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker")
        self.fastpitch.eval()

        self.hifigan = HifiGanModel.from_pretrained(
            "tts_en_hifitts_hifigan_ft_fastpitch"
        )
        self.hifigan.eval()

        self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.speaker_encoder.eval()

        self.speaker_projection = SpeakerProjection(input_dim=192, output_dim=384)

        device = next(self.fastpitch.parameters()).device
        self.speaker_projection = self.speaker_projection.to(device)

        if verbose:
            print("✅ Models loaded")
            print(f"   FastPitch: Multi-speaker (384D embeddings)")
            print(f"   TitaNet: Speaker encoder (192D)")
            print(f"   Projection: 192D → 384D (learned)")

    def extract_speaker_embedding(self, audio_path, target_sr=16000):
        if self.verbose:
            print(f"Extracting speaker embedding from: {audio_path}")

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
            print(f"✅ Speaker embedding extracted: {embedding.shape}")

        return embedding

    def inject_speaker_embedding(self, speaker_embedding_192d, speaker_id=0):
        if self.verbose:
            print(f"Injecting speaker embedding into slot {speaker_id}...")

        with torch.no_grad():
            projected_embedding = self.speaker_projection(speaker_embedding_192d)

        projected_embedding = projected_embedding / torch.norm(projected_embedding)

        speaker_embeddings = self.fastpitch.fastpitch.speaker_emb.weight.data
        speaker_embeddings[speaker_id] = projected_embedding

        if self.verbose:
            print(f"✅ Embedding projected and injected")
            print(f"   192D → 384D via learned projection")
            print(f"   Replaced speaker {speaker_id}")

    def generate(self, text, speaker_id=0, pace=1.0, output="output.wav"):
        if self.verbose:
            print(f"Generating speech (speaker {speaker_id}, pace {pace}x)...")

        tokens = self.fastpitch.parse(text)
        spec = self.fastpitch.generate_spectrogram(
            tokens=tokens, speaker=speaker_id, pace=pace
        )

        audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().detach().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, 44100)

        if self.verbose:
            print(f"✅ Saved: {output} ({len(audio) / 44100:.2f}s)")

        return output

    def train_projection(self, voice_samples, num_epochs=100, lr=0.001):
        print(f"\n🎓 Training speaker projection layer...")
        print(f"   Samples: {len(voice_samples)}")
        print(f"   Epochs: {num_epochs}")

        embeddings = []
        for sample in voice_samples:
            emb = self.extract_speaker_embedding(sample)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)

        reference_embeddings = self.fastpitch.fastpitch.speaker_emb.weight.data
        reference_mean = reference_embeddings.mean(dim=0)
        reference_std = reference_embeddings.std(dim=0)

        optimizer = torch.optim.Adam(self.speaker_projection.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            projected = self.speaker_projection(embeddings)

            loss_mean = torch.mean((projected.mean(dim=0) - reference_mean) ** 2)
            loss_std = torch.mean((projected.std(dim=0) - reference_std) ** 2)
            loss = loss_mean + loss_std

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.6f}")

        print(f"✅ Projection layer trained!")


if __name__ == "__main__":
    import sys

    tts = FastPitchWithSpeakerProjection()

    if len(sys.argv) < 2:
        print("Usage: python fastpitch_projection.py <voice_audio.wav> [text]")
        print("\nExample:")
        print('  python fastpitch_projection.py custom-voice/record.wav "Hello world"')
        sys.exit(1)

    voice_file = sys.argv[1]
    text = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "This is a test with projected speaker embedding."
    )

    embedding = tts.extract_speaker_embedding(voice_file)

    tts.inject_speaker_embedding(embedding, speaker_id=0)

    tts.generate(text, speaker_id=0, output="output_projected.wav")

    print("\n💡 Note: The projection layer is initialized randomly.")
    print("   For best results, train it using tts.train_projection([audio_files])")
