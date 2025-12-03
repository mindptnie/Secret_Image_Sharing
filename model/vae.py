# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor

from typing import List

import graph as plot_graphs

class VAE(nn.Module):
    def __init__(self, 
                 image_size: int, 
                 latent_dim: int , 
                 num_channels: int =1) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        self.final_conv_size = image_size // 8
        self.final_feature_dim = 512 * self.final_conv_size * self.final_conv_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [batch, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # -> [batch, 512, 32, 32]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.mu_layer = nn.Linear(self.final_feature_dim, latent_dim)
        self.log_var_layer = nn.Linear(self.final_feature_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, self.final_feature_dim)
        
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu:Tensor, log_var:Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: Tensor) -> List[Tensor]:
        encoded = self.encoder(x)
        encoded_flat = encoded.view(-1, self.final_feature_dim)
        
        mu = self.mu_layer(encoded_flat)
        log_var = self.log_var_layer(encoded_flat)
        
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 512, self.final_conv_size, self.final_conv_size)
        
        reconstructed = self.decoder(decoder_input)
        return reconstructed

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        return [self.decode(z), x, mu, log_var]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        recon_loss = F.mse_loss(recons, input) 
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        loss = recon_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'Reconstruction_Loss':recon_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        
        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
