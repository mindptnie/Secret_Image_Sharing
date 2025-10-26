import torch.exp as t_exp
import torch.randn_like as t_randn_like
import torch.nn as t_nn

import utility as util
import graph as plot_graphs

class VariationalAutoencoder(t_nn.Module):
    def __init__(self, image_size: int, latent_dim: int):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # 編碼器 (Encoder)
        self.encoder = t_nn.Sequential(
            t_nn.Linear(image_size, 2048),
            t_nn.ReLU(),
            t_nn.Linear(2048, 1024),
            t_nn.ReLU(),
        )

        # 解碼器 (Decoder)
        self.decoder = t_nn.Sequential(
            t_nn.Linear(latent_dim, 1024),
            t_nn.ReLU(),
            t_nn.Linear(1024, 2048),
            t_nn.ReLU(),
            t_nn.Linear(2048, image_size),
            t_nn.Sigmoid(),  # 限制輸出在 [0,1]
        )

        self.mu_layer = t_nn.Linear(1024, latent_dim)  # 平均值 (mu)
        self.log_var_layer = t_nn.Linear(1024, latent_dim)  # log(方差) (log_var)
        pass

    def reparameterize(self, mu, log_var):
        std = t_exp(0.5 * log_var)  # 計算標準差
        eps = t_randn_like(std)  # 標準正態分布的隨機數
        return mu + eps * std  # reparameterization trick

    def forward(self, x):
        x = x.view(-1, self.image_size)  # 攤平成 1D
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        #新增
        # **確保輸出 shape 為 `[batch_size, 1, 128, 128]`**
        reconstructed = reconstructed.view(-1, 1, 128, 128)
        return mu, log_var, z, reconstructed
    
    # 訓練 VAE (含學習率調度)
    def train_vae(self, optimizer, scheduler, train_loader, epochs, device):
        loss_history = []
        lr_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            for data in train_loader:
                #img = data.view(-1, 1, 128, 128)  # 確保輸入形狀正確
                #img = data.view(-1, 1, 128, 128).to(device)  # 把資料搬到 GPU
                img = data.view(-1, 1, 256, 256).to(device)  # 把資料搬到 GPU

                optimizer.zero_grad()
                #mu, log_var, latent, reconstructed = vae(data)
                mu, log_var, latent, reconstructed = self.forward(img) # 確保在 GPU 計算

                # **檢查 reconstructed 和 img 的 shape**
                assert reconstructed.shape == img.shape, f"Shape mismatch: {reconstructed.shape} vs {img.shape}"

                #loss = vae_loss_function(reconstructed, data, mu, log_var)
                loss = util.vae_loss_function(reconstructed, img, mu, log_var)  # 用 `img`，而不是 `data`
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # 記錄 epoch 平均 loss
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            # 更新學習率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
            print(f"Epoch {epoch+1}/{epochs},  Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

            #if (epoch + 1) % 5 == 0:
            #    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        plot_graphs.loss_curve(epochs, loss_history, output_dir="results")
        plot_graphs.learning_rate(epochs, lr_history, output_dir="results")
        
        pass