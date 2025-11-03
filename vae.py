import math
import time
from tqdm import tqdm
from torch import exp as t_exp
from torch import randn_like as t_randn_like
import torch.nn as t_nn
import torch

import utility as util
import graph as plot_graphs

class VariationalAutoencoder(t_nn.Module):
    def __init__(self, image_size: tuple, latent_dim: int):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Calculate the size after convolutions
        # For 256x256: 256 -> 128 -> 64 -> 32 -> 16
        self.final_conv_size = image_size[0] // 16
        self.final_feature_dim = 256 * self.final_conv_size * self.final_conv_size
        
        # Encoder - Convolutional layers
        self.encoder = t_nn.Sequential(
            # Input: [batch, 1, 256, 256]
            t_nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            t_nn.BatchNorm2d(32),
            t_nn.ReLU(),
            
            t_nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            t_nn.BatchNorm2d(64),
            t_nn.ReLU(),
            
            t_nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [batch, 128, 32, 32]
            t_nn.BatchNorm2d(128),
            t_nn.ReLU(),
            
            t_nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [batch, 256, 16, 16]
            t_nn.BatchNorm2d(256),
            t_nn.ReLU(),
        )
        
        # Latent space layers
        self.mu_layer = t_nn.Linear(self.final_feature_dim, latent_dim)
        self.log_var_layer = t_nn.Linear(self.final_feature_dim, latent_dim)
        
        # Projection from latent to decoder input
        self.decoder_input = t_nn.Linear(latent_dim, self.final_feature_dim)
        
        # Decoder - Transposed Convolutional layers
        self.decoder = t_nn.Sequential(
            # Input: [batch, 256, 16, 16]
            t_nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [batch, 128, 32, 32]
            t_nn.BatchNorm2d(128),
            t_nn.ReLU(),
            
            t_nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [batch, 64, 64, 64]
            t_nn.BatchNorm2d(64),
            t_nn.ReLU(),
            
            t_nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 128, 128]
            t_nn.BatchNorm2d(32),
            t_nn.ReLU(),
            
            t_nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> [batch, 1, 256, 256]
            t_nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = t_exp(0.5 * log_var)
        eps = t_randn_like(std)
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

                img = data.view(-1, 1, self.image_size[0], self.image_size[1]).to(device)

                optimizer.zero_grad()
                mu, log_var, latent, reconstructed = self.forward(img)

                loss = util.vae_loss_function(reconstructed, img, mu, log_var, lambda_weight=lambda_weight)
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