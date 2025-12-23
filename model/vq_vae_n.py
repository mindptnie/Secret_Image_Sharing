import time
from tqdm import tqdm

from .base import BaseVAE

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor

from typing import List

import graph as plot_graphs


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


class VQ_VAE(BaseVAE):
    def __init__(self, 
                 image_size: int, 
                 codebook_size: int = 512,  # Codebook size
                 codebook_dim: int = 64,    # Dimension of each code
                 num_channels: int = 1,
                 commitment_cost: float = 0.25) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_channels = num_channels
        
        # Calculate the size after convolutions
        # For 256x256: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.final_conv_size = image_size // 32 
        
        self.encoder = nn.Sequential(
            # 1. Downsample (256 -> 128)
            nn.Conv2d(self.num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 2. Downsample (128 -> 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 3. Downsample (64 -> 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 4. Keep Size (32 -> 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 5. Keep Size (32 -> 32)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.pre_quantization_conv = nn.Conv2d(512, codebook_dim, kernel_size=1)
        
        self.vector_quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost
        )
        
        self.post_quantization_conv = nn.Conv2d(codebook_dim, 512, kernel_size=1)
        
        # Decoder (Mirror Encoder)
        self.decoder = nn.Sequential(
            # Input: [batch, 512, 32, 32]
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x: Tensor) -> tuple:
        """
        Encode input and get quantized representation
        """
        encoded = self.encoder(x)  
        z_e = self.pre_quantization_conv(encoded)  
        return z_e
    
    def decode(self, z_q: Tensor) -> Tensor:
        """
        Decode from quantized representation
        """
        z = self.post_quantization_conv(z_q)  
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
        Calculate VQ-VAE loss with L1 + MSE
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        
        # 1. MSE Loss (เก็บไว้คุมโครงสร้างภาพรวม เพื่อความเสถียร)
        mse_loss = F.mse_loss(recons, input)
        
        # 2. L1 Loss (ตัวช่วยเรื่องสีและความคมชัด)
        # คำนวณความต่างแบบ Absolute ซึ่งจะลงโทษสีที่เพี้ยนหนักกว่า MSE
        l1_loss = F.l1_loss(recons, input)
        
        # กำหนดน้ำหนักของ L1 (ปรับได้)
        # แนะนำให้เริ่มที่ 1.0 ถ้าสียังจืดให้เพิ่มเป็น 2.0 หรือ 5.0
        l1_weight = 1.0 
        
        # Total loss: ผสม MSE และ L1 เข้าด้วยกัน
        loss = mse_loss + (l1_weight * l1_loss) + vq_loss
        
        return {
            'loss': loss, 
            'MSE_Loss': mse_loss.detach(), 
            'L1_Loss': l1_loss.detach(),    # เอาไว้ดูใน Graph ว่าลดลงไหม
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
        _, _, encoding_indices = self.vector_quantizer(z_e)
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