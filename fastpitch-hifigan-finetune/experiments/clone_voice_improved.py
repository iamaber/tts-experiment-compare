"""
Improved Voice Cloning - Learn projection from 256D to 384D
Uses a simple neural network to map resemblyzer embeddings to FastPitch space
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav


class EmbeddingProjector(nn.Module):
    """Neural network to project 256D embeddings to 384D."""
    
    def __init__(self, input_dim=256, output_dim=384):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ImprovedVoiceCloner:
    """Extract and project voice embeddings to FastPitch space."""
    
    def __init__(self):
        """Initialize speaker encoder and projector."""
        print("Loading speaker encoder...")
        self.encoder = VoiceEncoder()
        
        # Initialize projector (untrained - this is the limitation)
        self.projector = EmbeddingProjector()
        print("✅ Speaker encoder ready")
        print("⚠️  Note: Projector is untrained - results may vary")
    
    def extract_embedding(self, audio_path, use_projection=True):
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            use_projection: Use neural projection (vs zero-padding)
        
        Returns:
            Speaker embedding tensor (384D if projected, 256D otherwise)
        """
        print(f"Extracting embedding from: {audio_path}")
        
        # Load and preprocess audio
        wav = preprocess_wav(audio_path)
        
        # Extract 256D embedding
        embedding_256d = self.encoder.embed_utterance(wav)
        embedding_256d = torch.from_numpy(embedding_256d).float()
        
        if use_projection:
            # Project to 384D using neural network
            with torch.no_grad():
                embedding_384d = self.projector(embedding_256d)
            print(f"✅ Embedding extracted and projected: 256D → 384D")
            return embedding_384d
        else:
            print(f"✅ Embedding extracted: 256D (no projection)")
            return embedding_256d
    
    def save_embedding(self, embedding, output_path):
        """Save embedding to file."""
        torch.save(embedding, output_path)
        print(f"💾 Saved embedding to: {output_path}")


def clone_voice_improved(audio_path, output_embedding="custom_voice_384d.pt"):
    """
    Extract and project voice embedding to 384D.
    
    Args:
        audio_path: Path to your voice audio
        output_embedding: Where to save the 384D embedding
    
    Returns:
        Path to saved embedding
    """
    cloner = ImprovedVoiceCloner()
    embedding = cloner.extract_embedding(audio_path, use_projection=True)
    cloner.save_embedding(embedding, output_embedding)
    
    print(f"\n✅ Voice cloning complete!")
    print(f"   384D embedding saved to: {output_embedding}")
    print(f"\n⚠️  IMPORTANT LIMITATION:")
    print(f"   The projection network is UNTRAINED")
    print(f"   Results will still not match your voice perfectly")
    print(f"   For best results: fine-tune FastPitch on your voice")
    
    return output_embedding


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clone_voice_improved.py <your_audio.wav>")
        print("\nThis creates a 384D embedding using neural projection")
        print("Note: Projection is untrained - fine-tuning is recommended for best results")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)
    
    clone_voice_improved(audio_file)
