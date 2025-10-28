import math
import time
from tqdm import tqdm
from torch import exp as t_exp
from torch import randn_like as t_randn_like
import torch.nn as t_nn

import utility as util
import graph as plot_graphs

class VariationalAutoencoder(t_nn.Module):
    def __init__(self, image_size:tuple, latent_dim: int):
        super().__init__()
        self.image_size = image_size
        # 編碼器 (Encoder)
        self.encoder = t_nn.Sequential(
            t_nn.Linear(image_size[0] * image_size[1], 2048),
            t_nn.ReLU(),
            t_nn.Linear(2048, 1024),
            t_nn.ReLU(),
        )
        
        self.mu_layer = t_nn.Linear(1024, latent_dim)  # 平均值 (mu)
        self.log_var_layer = t_nn.Linear(1024, latent_dim)  # log(方差) (log_var)

        # 解碼器 (Decoder)
        self.decoder = t_nn.Sequential(
            t_nn.Linear(latent_dim, 1024),
            t_nn.ReLU(),
            t_nn.Linear(1024, 2048),
            t_nn.ReLU(),
            t_nn.Linear(2048, image_size[0] * image_size[1]),
            t_nn.Sigmoid(),  # 限制輸出在 [0,1]
        )

        pass

    def reparameterize(self, mu, log_var):
        std = t_exp(0.5 * log_var)  # 計算標準差
        eps = t_randn_like(std)  # 標準正態分布的隨機數
        return mu + eps * std  # reparameterization trick

    def forward(self, x):
        x = x.view(-1, self.image_size[0] * self.image_size[1])  # 攤平成 1D
        
        encoded = self.encoder(x)
        
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        
        z = self.reparameterize(mu, log_var)
        
        reconstructed = self.decoder(z)
        
        #新增
        # **確保輸出 shape 為 `[batch_size, 1, 128, 128]`**
        reconstructed = reconstructed.view(-1, 1, self.image_size[0], self.image_size[1])
        # reconstructed = reconstructed.view(-1, 1, 256, 256)
        return mu, log_var, z, reconstructed
    
    # 訓練 VAE (含學習率調度)
    def train_vae(self, optimizer, scheduler, train_loader, epochs, device):
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

                loss = util.vae_loss_function(reconstructed, img, mu, log_var) 
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

        pass