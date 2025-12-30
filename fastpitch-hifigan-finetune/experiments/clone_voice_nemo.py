"""
Voice Cloning using NVIDIA NeMo's TitaNet Speaker Encoder
Uses NeMo's own speaker verification model for better compatibility
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class NeMoVoiceCloner:
    """Extract speaker embeddings using NVIDIA's TitaNet model."""
    
    def __init__(self):
        """Initialize NVIDIA TitaNet speaker encoder."""
        print("Loading NVIDIA TitaNet speaker encoder...")
        
        # Load NeMo's speaker verification model
        self.encoder = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        self.encoder.eval()
        
        print(f"✅ TitaNet loaded - embedding dimension: {self.encoder.cfg.decoder.emb_sizes}")
    
    def extract_embedding(self, audio_path, target_sr=16000):
        """
        Extract speaker embedding from audio file using TitaNet.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (TitaNet uses 16kHz)
        
        Returns:
            Speaker embedding tensor
        """
        print(f"Extracting embedding from: {audio_path}")
        
        # Load audio using librosa
        import librosa
        waveform, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).float()
        
        # Resample if needed
        if sr != target_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=False)
        
        # TitaNet expects shape (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Get audio length
        audio_length = torch.tensor([waveform.shape[1]])
        
        # Move to same device as model
        device = next(self.encoder.parameters()).device
        waveform = waveform.to(device)
        audio_length = audio_length.to(device)
        
        # Extract embedding
        with torch.no_grad():
            # Get embeddings from TitaNet
            _, embeddings = self.encoder(input_signal=waveform, input_signal_length=audio_length)
            embedding = embeddings.squeeze().cpu()  # Move back to CPU for saving
        
        print(f"✅ Embedding extracted: shape {embedding.shape}")
        return embedding
    
    def save_embedding(self, embedding, output_path):
        """Save embedding to file."""
        torch.save(embedding, output_path)
        print(f"💾 Saved embedding to: {output_path}")


def clone_voice_nemo(audio_path, output_embedding="custom_voice_nemo.pt"):
    """
    Extract speaker embedding using NVIDIA's TitaNet model.
    
    Args:
        audio_path: Path to your voice audio
        output_embedding: Where to save the embedding
    
    Returns:
        Path to saved embedding
    """
    cloner = NeMoVoiceCloner()
    embedding = cloner.extract_embedding(audio_path)
    cloner.save_embedding(embedding, output_embedding)
    
    print(f"\n✅ Voice cloning complete!")
    print(f"   Embedding saved to: {output_embedding}")
    print(f"   Using NVIDIA TitaNet speaker encoder")
    print(f"\n💡 Next step: Use this embedding with custom_tts.py")
    
    return output_embedding


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clone_voice_nemo.py <your_audio.wav>")
        print("\nThis uses NVIDIA's TitaNet speaker encoder")
        print("Better compatibility with NeMo FastPitch")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)
    
    clone_voice_nemo(audio_file)
