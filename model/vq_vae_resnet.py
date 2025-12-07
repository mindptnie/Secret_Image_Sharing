import time
from tqdm import tqdm

from .base import BaseVAE

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor

from typing import List

import graph as plot_graphs


class Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        # Block: Conv3x3 -> ReLU -> Conv1x1 (bottleneck structure)
        self._block = nn.Sequential(
            nn.ReLU(),
            # 1. Bottleneck Conv (reduces channels)
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            # 2. Final Conv (restores channels)
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
        
        # Ensure num_hiddens is the output channel count
        self._input_is_different = in_channels != num_hiddens
        if self._input_is_different:
             self._input_projection = nn.Conv2d(in_channels, num_hiddens, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Standard ResNet: Output = x + F(x)
        # If input channels and output channels are different, project input first
        if self._input_is_different:
            identity = self._input_projection(x)
        else:
            identity = x
            
        return identity + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        # Create a list of Residual blocks
        self._layers = nn.ModuleList([
            Residual(in_channels if i == 0 else num_hiddens, num_hiddens, num_residual_hiddens)
            for i in range(self._num_residual_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        # Apply final ReLU after the stack
        return F.relu(x)
    
class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that replaces continuous latent space with discrete codebook
    """
    def __init__(self, codebook_size: int, codebook_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size  # Size of codebook
        self.codebook_dim = codebook_dim    # Dimension of each embedding
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
    def forward(self, z: Tensor) -> tuple:
        """
        Args:
            z: Encoder output [batch, codebook_dim, height, width]
        Returns:
            quantized: Quantized version [batch, codebook_dim, height, width]
            loss: VQ loss (commitment + codebook)
            encoding_indices: Codebook indices [batch, height, width]
        """
        # Convert from [B, C, H, W] to [B, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)  # [B*H*W, C]
        
        # Calculate distances to codebook vectors
        # (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find closest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(z.shape)
        
        # Calculate VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Convert back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Reshape encoding indices
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        
        return quantized, loss, encoding_indices

class VQ_VAE_ResNet(BaseVAE):
    def __init__(self, 
                 image_size: int, 
                 codebook_size: int = 512,  # Codebook size
                 codebook_dim: int = 64,    # Dimension of each code
                 num_channels: int = 1,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 # --- NEW HYPERPARAMETERS for Residual Blocks ---
                 num_residual_layers: int = 2,    # Number of residual blocks in the stack
                 num_residual_hiddens: int = 32) -> None: # Channels in the internal bottleneck
        super().__init__()
        
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_channels = num_channels
        
        # Calculate the size after convolutions
        self.final_conv_size = image_size // 32 
        
        # Encoder - Convolutional layers (same as before)
        self.encoder = nn.Sequential(
            # Input: [batch, 1, 256, 256]
            nn.Conv2d(self.num_channels, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [batch, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # -> [batch, 512, 32, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            ResidualStack(512, 512, num_residual_layers, num_residual_hiddens)
        )
        
        # Project to embedding dimension for quantization
        self.pre_quantization_conv = nn.Conv2d(512, codebook_dim, kernel_size=1)
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )
        
        # Project back from embedding dimension
        self.post_quantization_conv = nn.Conv2d(codebook_dim, 512, kernel_size=1)
        
        # Decoder - Transposed Convolutional layers
        self.decoder = nn.Sequential(
            # Input: [batch, 512, 32, 32]
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), # -> [batch, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # ResidualStack should output 256 channels to match the next layer
            ResidualStack(256, 256, num_residual_layers, num_residual_hiddens),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # -> [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # -> [batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=2, padding=1),  # -> [batch, 1, 256, 256]
            nn.Sigmoid(),
        )
    
    def encode(self, x: Tensor) -> tuple:
        """
        Encode input and get quantized representation
        """
        encoded = self.encoder(x)  # [batch, 512, 8, 8]
        z_e = self.pre_quantization_conv(encoded)  # [batch, codebook_dim, 8, 8]
        return z_e
    
    def decode(self, z_q: Tensor) -> Tensor:
        """
        Decode from quantized representation
        """
        z = self.post_quantization_conv(z_q)  # [batch, 512, 8, 8]
        reconstructed = self.decoder(z)
        return reconstructed

    def forward(self, x: Tensor) -> List[Tensor]:
        # Encode
        z_e = self.encode(x)
        
        # Quantize
        z_q, vq_loss, encoding_indices = self.vector_quantizer(z_e)
        
        # Decode
        reconstructed = self.decode(z_q)
        
        return [reconstructed, x, vq_loss, encoding_indices]
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Calculate VQ-VAE loss
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recons, input)
        
        # Total loss
        loss = recon_loss + vq_loss
        
        return {
            'loss': loss, 
            'Reconstruction_Loss': recon_loss.detach(), 
            'VQ_Loss': vq_loss.detach()
        }
    
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """
        Sample from the codebook randomly
        """
        # Randomly sample codebook indices
        indices = torch.randint(
            0, self.codebook_size, 
            (num_samples, self.final_conv_size, self.final_conv_size),
            device=device
        )
        
        # Convert indices to one-hot
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        
        # Get embeddings
        quantized = torch.matmul(
            one_hot.view(-1, self.codebook_size),
            self.vector_quantizer.embedding.weight
        )
        
        # Reshape to [batch, height, width, codebook_dim]
        quantized = quantized.view(
            num_samples, 
            self.final_conv_size, 
            self.final_conv_size, 
            self.codebook_dim
        )
        
        # Convert to [batch, codebook_dim, height, width]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        samples = self.decode(quantized)
        return samples
    
    def generate(self, x: Tensor) -> Tensor:
        """
        Generate reconstruction
        """
        return self.forward(x)[0]
    
    def get_codebook_indices(self, x: Tensor) -> Tensor:
        """
        Get the codebook indices for an input
        """
        z_e = self.encode(x)
        _, _, encoding_indices= self.vector_quantizer(z_e)
        return encoding_indices
    
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """
        Decode from codebook indices
        Args:
            indices: [batch, height, width] codebook indices
        """
        # Convert indices to one-hot
        one_hot = F.one_hot(indices.long(), num_classes=self.codebook_size).float()
        
        # Get embeddings
        quantized = torch.matmul(
            one_hot.view(-1, self.codebook_size),
            self.vector_quantizer.embedding.weight
        )
        
        # Reshape to [batch, height, width, codebook_dim]
        batch_size, height, width = indices.shape
        quantized = quantized.view(batch_size, height, width, self.codebook_dim)
        
        # Convert to [batch, codebook_dim, height, width]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        return self.decode(quantized)