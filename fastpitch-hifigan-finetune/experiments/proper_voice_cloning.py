import torch
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class ProperVoiceCloning:
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

        if verbose:
            print("✅ All models loaded")
            print(f"   FastPitch: Single-speaker ({self.hifigan.sample_rate} Hz)")
            print(f"   HiFiGAN: Vocoder")
            print(f"   TitaNet: Speaker encoder (192D)")

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
            embedding = embeddings.squeeze().cpu()

        if self.verbose:
            print(f"✅ Speaker embedding extracted: {embedding.shape}")

        return embedding

    def generate(self, text, pace=1.0, output="output.wav"):
        if self.verbose:
            print(f"Generating speech...")

        tokens = self.fastpitch.parse(text)

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


if __name__ == "__main__":
    import sys

    tts = ProperVoiceCloning()

    if len(sys.argv) > 1:
        voice_file = sys.argv[1]
        embedding = tts.extract_speaker_embedding(voice_file)

        torch.save(embedding, "speaker_embedding.pt")
        print(f"\n💾 Saved speaker embedding to: speaker_embedding.pt")

    text = "This is a test of the single-speaker FastPitch model."
    if len(sys.argv) > 2:
        text = sys.argv[2]

    tts.generate(text, output="output_single_speaker.wav")

    print("\n⚠️  IMPORTANT NOTE:")
    print("   Single-speaker FastPitch doesn't support speaker embeddings directly")
    print("   The model needs to be fine-tuned on your voice for true voice cloning")
    print("   This script demonstrates the architecture but won't clone your voice yet")
