"""
Custom Voice Cloning for FastPitch
Extract speaker embedding from your audio and replace a speaker in the model
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav


class VoiceCloner:
    """Extract and inject custom voice embeddings into FastPitch."""
    
    def __init__(self):
        """Initialize speaker encoder."""
        print("Loading speaker encoder...")
        self.encoder = VoiceEncoder()
        print("✅ Speaker encoder ready")
    
    def extract_embedding(self, audio_path):
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
        
        Returns:
            Speaker embedding tensor
        """
        print(f"Extracting embedding from: {audio_path}")
        
        # Load and preprocess audio
        wav = preprocess_wav(audio_path)
        
        # Extract embedding
        embedding = self.encoder.embed_utterance(wav)
        
        # Convert to torch tensor
        embedding = torch.from_numpy(embedding).float()
        
        print(f"✅ Embedding extracted: shape {embedding.shape}")
        return embedding
    
    def save_embedding(self, embedding, output_path):
        """Save embedding to file."""
        torch.save(embedding, output_path)
        print(f"💾 Saved embedding to: {output_path}")
    
    def load_embedding(self, embedding_path):
        """Load embedding from file."""
        embedding = torch.load(embedding_path)
        print(f"📥 Loaded embedding from: {embedding_path}")
        return embedding


def clone_voice(audio_path, output_embedding="custom_voice.pt"):
    """
    Extract speaker embedding from your voice.
    
    Args:
        audio_path: Path to your voice audio (10-30 seconds recommended)
        output_embedding: Where to save the embedding
    
    Returns:
        Path to saved embedding
    """
    cloner = VoiceCloner()
    embedding = cloner.extract_embedding(audio_path)
    cloner.save_embedding(embedding, output_embedding)
    
    print(f"\n✅ Voice cloning complete!")
    print(f"   Embedding saved to: {output_embedding}")
    print(f"\n💡 Next step: Use this embedding with custom_tts.py")
    
    return output_embedding


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clone_voice.py <your_audio.wav>")
        print("\nExample:")
        print("  python clone_voice.py my_voice.wav")
        print("\nThis will create 'custom_voice.pt' with your voice embedding")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)
    
    clone_voice(audio_file)
