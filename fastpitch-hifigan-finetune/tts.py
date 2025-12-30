"""
Multi-Speaker TTS Library using NVIDIA NeMo
Simple class-based interface - no CLI needed
"""

from pathlib import Path
import soundfile as sf
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


class TTS:
    """Simple multi-speaker TTS interface."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", verbose=True):
        """
        Initialize and load TTS models.
        
        Args:
            device: 'cuda' or 'cpu'
            verbose: Print loading messages
        """
        self.device = device
        self.verbose = verbose
        
        if verbose:
            print("Loading models...")
        
        # Load models
        self.fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker")
        self.fastpitch = self.fastpitch.to(device)
        self.fastpitch.eval()
        
        self.hifigan = HifiGanModel.from_pretrained("tts_en_hifitts_hifigan_ft_fastpitch")
        self.hifigan = self.hifigan.to(device)
        self.hifigan.eval()
        
        self.n_speakers = self.fastpitch.cfg.n_speakers
        
        if verbose:
            print(f"Ready: {self.n_speakers} speakers available")
    
    def generate(self, text, speaker=92, pace=1.0, output="output.wav"):
        """
        Generate speech and save to file.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (0-12799, default: 92)
            pace: Speed (0.1-2.0, default: 1.0)
            output: Output file path
        
        Returns:
            Path to saved audio file
        """
        # Validate
        pace = max(0.1, min(2.0, pace))
        if speaker >= self.n_speakers:
            raise ValueError(f"Speaker {speaker} out of range (0-{self.n_speakers-1})")
        
        # Synthesize
        with torch.no_grad():
            tokens = self.fastpitch.parse(text)
            spec = self.fastpitch.generate_spectrogram(tokens=tokens, speaker=speaker, pace=pace)
            audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Save
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, 44100)
        
        if self.verbose:
            print(f"✅ Saved: {output} ({len(audio)/44100:.2f}s)")
        
        return output


# Quick usage example
if __name__ == "__main__":
    # Initialize
    tts = TTS()
    
    # Generate speech
    tts.generate("However, this modeling complexity comes at a computational cost. Despite being non-autoregressive, VITS still requires heavier inference computation than FastPitch, which explains its higher latency and RTF. While it remains fast enough for real-time usage, it is less suitable for extremely high-throughput systems or environments with strict GPU constraints.", speaker=92, pace=1.0 , output="test.wav")
    
    # Batch generation
    # for i, text in enumerate(["First", "Second", "Third"]):
    #     tts.generate(text, speaker=92, pace=1.0, output=f"batch_{i}.wav")
