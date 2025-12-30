"""
Proper Voice Cloning with Single-Speaker FastPitch + TitaNet
Uses nvidia/tts_en_fastpitch (single-speaker) with TitaNet speaker embeddings
"""

import torch
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class ProperVoiceCloning:
    """Voice cloning using single-speaker FastPitch with TitaNet embeddings."""
    
    def __init__(self, verbose=True):
        """Initialize models."""
        self.verbose = verbose
        
        if verbose:
            print("Loading models...")
        
        # Load single-speaker FastPitch (not multispeaker!)
        self.fastpitch = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
        self.fastpitch.eval()
        
        # Load HiFiGAN vocoder
        self.hifigan = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
        self.hifigan.eval()
        
        # Load TitaNet speaker encoder
        self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.speaker_encoder.eval()
        
        if verbose:
            print("✅ All models loaded")
            print(f"   FastPitch: Single-speaker (22050 Hz)")
            print(f"   HiFiGAN: Vocoder")
            print(f"   TitaNet: Speaker encoder (192D)")
    
    def extract_speaker_embedding(self, audio_path, target_sr=16000):
        """Extract speaker embedding from audio using TitaNet."""
        if self.verbose:
            print(f"Extracting speaker embedding from: {audio_path}")
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform).float()
        
        # Resample if needed
        if sr != target_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Prepare for TitaNet
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        audio_length = torch.tensor([waveform.shape[1]])
        
        # Move to device
        device = next(self.speaker_encoder.parameters()).device
        waveform = waveform.to(device)
        audio_length = audio_length.to(device)
        
        # Extract embedding
        with torch.no_grad():
            _, embeddings = self.speaker_encoder(
                input_signal=waveform,
                input_signal_length=audio_length
            )
            embedding = embeddings.squeeze().cpu()
        
        if self.verbose:
            print(f"✅ Speaker embedding extracted: {embedding.shape}")
        
        return embedding
    
    def generate(self, text, speaker_embedding=None, pace=1.0, output="output.wav"):
        """
        Generate speech with optional speaker embedding.
        
        Args:
            text: Text to synthesize
            speaker_embedding: TitaNet embedding (192D) or None for default voice
            pace: Speaking speed
            output: Output file path
        
        Returns:
            Path to generated audio
        """
        if self.verbose:
            print(f"Generating speech...")
        
        # Parse text
        tokens = self.fastpitch.parse(text)
        
        # Generate spectrogram
        # Note: Single-speaker FastPitch doesn't have speaker parameter
        # We would need to modify the model to accept speaker embeddings
        # For now, this generates with the default trained voice
        spec = self.fastpitch.generate_spectrogram(tokens=tokens, pace=pace)
        
        # Convert to audio
        audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().detach().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Save
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, 22050)  # Single-speaker FastPitch uses 22050 Hz
        
        if self.verbose:
            print(f"✅ Saved: {output} ({len(audio)/22050:.2f}s)")
        
        return output


if __name__ == "__main__":
    import sys
    
    # Initialize
    tts = ProperVoiceCloning()
    
    # Extract speaker embedding from your voice
    if len(sys.argv) > 1:
        voice_file = sys.argv[1]
        embedding = tts.extract_speaker_embedding(voice_file)
        
        # Save embedding
        torch.save(embedding, "speaker_embedding.pt")
        print(f"\n💾 Saved speaker embedding to: speaker_embedding.pt")
    
    # Generate speech
    text = "This is a test of the single-speaker FastPitch model."
    if len(sys.argv) > 2:
        text = sys.argv[2]
    
    tts.generate(text, output="output_single_speaker.wav")
    
    print("\n⚠️  IMPORTANT NOTE:")
    print("   Single-speaker FastPitch doesn't support speaker embeddings directly")
    print("   The model needs to be fine-tuned on your voice for true voice cloning")
    print("   This script demonstrates the architecture but won't clone your voice yet")
