import time
from tqdm import tqdm
from .base import BaseVAE
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from typing import List

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for VQGAN/GAN.
    Outputs a HxW grid of predictions instead of a single prediction.
    """
    def __init__(self, in_channels: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
            
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels
        
        # First layer (No BatchNorm)
        self.conv_blocks.append(nn.Sequential(
            nn.Conv2d(current_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        current_channels = hidden_dims[0]
        
        # Intermediate layers
        for next_channels in hidden_dims[1:]:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            current_channels = next_channels

        # Final output layer (outputs a single channel prediction map)
        self.output_layer = nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.conv_blocks:
            x = block(x)
        return self.output_layer(x) # Output: [B, 1, H', W']

# ==============================================================================
# 2. BASE VQ-VAE COMPONENTS (Residual, ResidualStack, VectorQuantizer)
# ==============================================================================

class Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
        self._input_is_different = in_channels != num_hiddens
        if self._input_is_different:
             self._input_projection = nn.Conv2d(in_channels, num_hiddens, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if self._input_is_different:
            identity = self._input_projection(x)
        else:
            identity = x
        return identity + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([
            Residual(in_channels if i == 0 else num_hiddens, num_hiddens, num_residual_hiddens)
            for i in range(self._num_residual_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z: Tensor) -> tuple:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim) 
        
        # Calculate distances (L2 norm)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find closest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
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
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        
        return quantized, loss, encoding_indices

# ==============================================================================
# 3. VQ_VAE_GAN MODEL (Main Class)
# ==============================================================================

class VQ_VAE_Gan(BaseVAE):
    def __init__(self, 
                 image_size: int, 
                 num_embeddings: int = 512, 
                 embedding_dim: int = 64, 
                 num_channels: int = 1,
                 commitment_cost: float = 0.25,
                 num_residual_layers: int = 2, 
                 num_residual_hiddens: int = 32,
                 gan_loss_weight: float = 0.1) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.gan_loss_weight = gan_loss_weight

        # Assumes total stride=4 (Adjust based on your final encoder strides)
        self.final_conv_size = image_size // 4 
        
        # --- Encoder --- (Simplified for brevity, based on your original structure)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512), nn.ReLU(),
            ResidualStack(512, 512, num_residual_layers, num_residual_hiddens)
        )
        
        self.pre_quantization_conv = nn.Conv2d(512, embedding_dim, kernel_size=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.post_quantization_conv = nn.Conv2d(embedding_dim, 512, kernel_size=1)
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            ResidualStack(512, 512, num_residual_layers, num_residual_hiddens),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid(),
        )
        
        # --- GAN COMPONENTS ---
        # The VAE part is now combined with GAN
        self.discriminator = Discriminator(in_channels=self.num_channels)
        
    def encode(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        z_e = self.pre_quantization_conv(encoded)
        return z_e
    
    def decode(self, z_q: Tensor) -> Tensor:
        z = self.post_quantization_conv(z_q)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def forward(self, x: Tensor) -> List[Tensor]:
        z_e = self.encode(x)
        z_q, vq_loss, encoding_indices = self.vector_quantizer(z_e)
        reconstructed = self.decode(z_q)
        return [reconstructed, x, vq_loss, encoding_indices]
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Calculates VQGAN Loss (Recon + Adversarial + VQ)
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2] # L_VQ (replaces KL-Divergence)
        
        # 1. Reconstruction Loss (L_Recon)
        recon_loss = F.mse_loss(recons, input) # Use MSE or L1
        
        # 2. Adversarial Loss (L_G) - Generator Loss (Target = 1)
        logits_fake = self.discriminator(recons)
        # Using Least Squares GAN (LSGAN) Loss
        gan_loss = F.mse_loss(logits_fake, torch.ones_like(logits_fake)) 
        
        # 3. Total Autoencoder Loss (L_AE)
        lambda_recon = 1.0 
        lambda_gan = self.gan_loss_weight # e.g., 0.1
        
        # L_AE = L_Recon + lambda_gan * L_G + L_VQ
        ae_loss = (lambda_recon * recon_loss + 
                   lambda_gan * gan_loss + 
                   vq_loss)
        
        output = {
            'loss': ae_loss, 
            'Recon_Loss': recon_loss.detach(), 
            'GAN_Adversarial_Loss': gan_loss.detach(),
            'VQ_Loss': vq_loss.detach()
        }
        
        # 4. Discriminator Loss (L_D) - (Target Real=1, Fake=0)
        if kwargs.get('optimize_discriminator', False):
            # L_D_Fake = MSE(D(x'), 0)
            logits_fake_detached = self.discriminator(recons.detach()) 
            d_loss_fake = F.mse_loss(logits_fake_detached, torch.zeros_like(logits_fake_detached))
            
            # L_D_Real = MSE(D(x), 1)
            logits_real = self.discriminator(input)
            d_loss_real = F.mse_loss(logits_real, torch.ones_like(logits_real))
            
            # L_D = 0.5 * (L_D_Fake + L_D_Real)
            d_loss = 0.5 * (d_loss_fake + d_loss_real)
            output['D_Loss'] = d_loss.detach()
        
        return output

    # --- Utility Methods (decode_from_indices, sample, generate, get_codebook_indices) ---
    # (These methods remain the same as your original code)
    
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        indices = torch.randint(
            0, self.num_embeddings, 
            (num_samples, self.final_conv_size, self.final_conv_size),
            device=device
        )
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        quantized = torch.matmul(
            one_hot.view(-1, self.num_embeddings),
            self.vector_quantizer.embedding.weight
        )
        quantized = quantized.view(
            num_samples, self.final_conv_size, self.final_conv_size, self.embedding_dim
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)
    
    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
    
    def get_codebook_indices(self, x: Tensor) -> Tensor:
        z_e = self.encode(x)
        _, _, encoding_indices= self.vector_quantizer(z_e)
        return encoding_indices
    
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        one_hot = F.one_hot(indices.long(), num_classes=self.num_embeddings).float()
        quantized = torch.matmul(
            one_hot.view(-1, self.num_embeddings),
            self.vector_quantizer.embedding.weight
        )
        batch_size, height, width = indices.shape
        quantized = quantized.view(batch_size, height, width, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)