import time
from tqdm import tqdm

import torch
from torch import nn
from torch import sum
from torch import mean
from torch.nn import functional as F
from torch import tensor as Tensor

import graph as plot_graphs

class VariationalAutoencoder(nn.Module):
    def __init__(self, 
                 image_size: int, 
                 latent_dim: int , 
                 num_channels: int =1) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        # Calculate the size after convolutions
        # For 256x256: 256 -> 128 -> 64 -> 32 -> 16
        self.final_conv_size = image_size // 16
        self.final_feature_dim = 256 * self.final_conv_size * self.final_conv_size
        
        # Encoder - Convolutional layers
        self.encoder = nn.Sequential(
            # Input: [batch, 1, 256, 256]
            nn.Conv2d(self.num_channels, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [batch, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Latent space layers
        self.mu_layer = nn.Linear(self.final_feature_dim, latent_dim)
        self.log_var_layer = nn.Linear(self.final_feature_dim, latent_dim)
        
        # Projection from latent to decoder input
        self.decoder_input = nn.Linear(latent_dim, self.final_feature_dim)
        
        # Decoder - Transposed Convolutional layers
        self.decoder = nn.Sequential(
            # Input: [batch, 256, 16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=2, padding=1),  # -> [batch, 1, 256, 256]
            nn.Sigmoid(),
        )

    def reparameterize(self, mu:Tensor, log_var:Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)  # [batch, 256, 16, 16]
        encoded_flat = encoded.view(-1, self.final_feature_dim)  # Flatten
        
        # Latent space
        mu = self.mu_layer(encoded_flat)
        log_var = self.log_var_layer(encoded_flat)
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 256, self.final_conv_size, self.final_conv_size)
        reconstructed = self.decoder(decoder_input)
        
        return mu, log_var, z, reconstructed
    
    def loss_function(reconstructed, original, mu, log_var, lambda_weight:float=0.0001):
        # **確保 reconstructed 的 shape 和 original 一樣**
        assert reconstructed.shape == original.shape, f"Shape mismatch: {reconstructed.shape} vs {original.shape}"

        recon_loss = nn.MSELoss()(reconstructed, original) # GPU 計算
        
        # 1. Sum over the latent dimensions (dim=1)
        kl_loss_per_item = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        # 2. Average across the batch
        kl_loss = mean(kl_loss_per_item) # GPU 計算
        
        return recon_loss + lambda_weight * kl_loss

    def train_vae(self, optimizer, scheduler, train_loader, epochs, device, lambda_weight: float = 0.0001):
        loss_history = []
        lr_history = []
        
        print("Starting training...")
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for data in progress_bar:
                if data is None:
                    continue

                img = data.to(device)

                optimizer.zero_grad()
                mu, log_var, latent, reconstructed = self.forward(img)

                loss = self.loss_function(reconstructed, img, mu, log_var, lambda_weight=lambda_weight)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                progress_bar.set_postfix(Loss=f"{epoch_loss:.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            print(f"Epoch {epoch+1}/{epochs} Summary: Avg Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

        progress_bar.close()
            
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Total training time: {total_time / 60:.2f} mins")
        
        print(f"epochs: {epochs}")
        print(f"loss_history: {loss_history}")
        print(f"lr_history: {lr_history}")

        plot_graphs.loss_curve(epochs, loss_history)
        plot_graphs.learning_rate(epochs, lr_history)