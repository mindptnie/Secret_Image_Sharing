import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from vae import VQVariationalAutoencoder

def test_vae_shape():
    print("Testing VAE shape...")
    try:
        model = VQVariationalAutoencoder(
            image_size=128, 
            num_embeddings=128,
            embedding_dim=64,
            num_channels=1
        )
        
        x = torch.randn(1, 1, 128, 128)
        z_e = model.encode(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Encoded shape: {z_e.shape}")
        
        if z_e.shape[2] == 32 and z_e.shape[3] == 32:
            print("SUCCESS: Latent shape is 32x32")
        else:
            print(f"FAILURE: Latent shape is {z_e.shape[2]}x{z_e.shape[3]}, expected 32x32")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vae_shape()
