#256*256版本 Latent Vector分組多項式秘密分享
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from PIL import Image
from Shamir1 import polynomial, decode
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 應該顯示 "cuda"


# 建立輸出資料夾
#output_dir = "VTest_output"
output_dir = "456_1_output"
os.makedirs(output_dir, exist_ok=True)

# 參數設定
latent_dim = 1024  # 潛在空間維度
epochs = 100
batch_size = 32
image_size = 256 * 256  
# 定義 Variational Autoencoder (VAE)
# 定義 Variational Autoencoder (VAE) latent_dim=1024
'''
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 編碼器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, latent_dim),  # 最後才壓縮成 1024 維
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(latent_dim, latent_dim)  # 平均值 (mu)
        self.log_var_layer = nn.Linear(latent_dim, latent_dim)  # log(方差) (log_var)

        # 解碼器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, image_size),  
            nn.Sigmoid()  # 限制範圍在 [0, 1]
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 計算標準差
        eps = torch.randn_like(std)  # 標準正態分布的隨機數
        return mu + eps * std  # reparameterization trick

    def forward(self, x):
        x = x.view(-1, image_size)  # 攤平成 1D
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        #新增
        # **確保輸出 shape 為 `[batch_size, 1, 128, 128]`**
        reconstructed = reconstructed.view(-1, 1, 256, 256)
        return mu, log_var, z, reconstructed
'''

'''
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 編碼器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, latent_dim)  # 平均值 (mu)
        self.log_var_layer = nn.Linear(256, latent_dim)  # log(方差) (log_var)

        # 解碼器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Sigmoid(),  # 限制輸出在 [0,1]
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 計算標準差
        eps = torch.randn_like(std)  # 標準正態分布的隨機數
        return mu + eps * std  # reparameterization trick

    def forward(self, x):
        x = x.view(-1, image_size)  # 攤平成 1D
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        #新增
        # **確保輸出 shape 為 `[batch_size, 1, 128, 128]`**
        reconstructed = reconstructed.view(-1, 1, 128, 128)
        return mu, log_var, z, reconstructed

'''
# 定義 Variational Autoencoder (VAE) latent_dim=1024
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 編碼器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(1024, latent_dim)  # 平均值 (mu)
        self.log_var_layer = nn.Linear(1024, latent_dim)  # log(方差) (log_var)

        # 解碼器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, image_size),
            nn.Sigmoid(),  # 限制輸出在 [0,1]
        )
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 計算標準差
        eps = torch.randn_like(std)  # 標準正態分布的隨機數
        return mu + eps * std  # reparameterization trick

    def forward(self, x):
        x = x.view(-1, image_size)  # 攤平成 1D
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        #新增
        # **確保輸出 shape 為 `[batch_size, 1, 128, 128]`**
        #reconstructed = reconstructed.view(-1, 1, 128, 128)
        reconstructed = reconstructed.view(-1, 1, 256, 256)
        return mu, log_var, z, reconstructed
  


# VAE 的損失函數 (重建誤差 + KL 散度)
def vae_loss_function(reconstructed, original, mu, log_var):
     # **確保 reconstructed 的 shape 和 original 一樣**
    assert reconstructed.shape == original.shape, f"Shape mismatch: {reconstructed.shape} vs {original.shape}"

    recon_loss = nn.MSELoss()(reconstructed, original) # GPU 計算
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL 散度  # GPU 計算
    return recon_loss + 0.0001 * kl_loss  # 調整 KL loss 權重

# 預處理影像
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join(output_dir, "grayscale_image.png"), img_resized)
    return ToTensor()(img_resized).view(-1, image_size).float()

# 自定義 Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return preprocess_image(self.image_paths[idx])
    

# Shamir's Secret Sharing 
def quantize_latent(latent, min_val=-3, max_val=3):
    latent_np = latent.detach().cpu().numpy()
    latent_scaled = (latent_np - min_val) / (max_val - min_val)
    latent_quantized = np.round(latent_scaled * 255).astype(np.uint8)
    return latent_quantized

def dequantize_latent(latent_quantized, min_val=-3, max_val=3):
    latent_scaled = latent_quantized.astype(np.float32) / 255.0
    latent = latent_scaled * (max_val - min_val) + min_val
    return latent

def create_shares(latent, n, r):
    latent_quantized = quantize_latent(latent)
    shares, shares_extra = polynomial(latent_quantized, n=n, r=r)

    shares_with_positions = []
    for i, share in enumerate(shares):
        share_tensor = torch.tensor(share, dtype=torch.int32)
        position = (i + 1,)  
        shares_with_positions.append((share_tensor, position, shares_extra[i]))
        if latent_dim == 256:
            share_image = share.reshape(16, 16).astype(np.uint8)
        elif latent_dim == 512:
            share_image = share.reshape(16, 32).astype(np.uint8)
        elif latent_dim == 1024:
            share_image = share.reshape(32, 32).astype(np.uint8)
        elif latent_dim == 1024:
            share_image = share.reshape(19, 18).astype(np.uint8)
        elif latent_dim == 2048:
            share_image = share.reshape(32, 64).astype(np.uint8)
        elif latent_dim == 4096:
            share_image = share.reshape(64, 64).astype(np.uint8)
        else:
            raise ValueError("Unsupported latent_dim for reshape")
        cv2.imwrite(os.path.join(output_dir, f"share_{i+1}.png"), share_image)

    #return shares_with_positions
    return shares_with_positions


def combine_shares(shares_with_positions, r):
    shares = np.array([share_tensor.numpy() for share_tensor, _, _ in shares_with_positions[:r]])
    shares_extra = [extra for _, _, extra in shares_with_positions[:r]]
    indices = [position[0] for _, position, _ in shares_with_positions[:r]]

    reconstructed_latent_quantized = decode(shares, shares_extra, indices, r=r)
    reconstructed_latent = dequantize_latent(reconstructed_latent_quantized)

    # ✅ 補 0 成 1024 維
    if latent_dim == 1024:
        pad_len = 1024 - reconstructed_latent.shape[0]
        reconstructed_latent = np.pad(reconstructed_latent, (0, pad_len), 'constant')

    return torch.tensor(reconstructed_latent, dtype=torch.float32)



# 訓練 VAE (含學習率調度)
def train_vae(vae, optimizer, scheduler, train_loader, epochs):

#def train_vae(vae, optimizer, train_loader, epochs):
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
            mu, log_var, latent, reconstructed = vae(img) # 確保在 GPU 計算

            # **檢查 reconstructed 和 img 的 shape**
            assert reconstructed.shape == img.shape, f"Shape mismatch: {reconstructed.shape} vs {img.shape}"

            #loss = vae_loss_function(reconstructed, data, mu, log_var)
            loss = vae_loss_function(reconstructed, img, mu, log_var)  # 用 `img`，而不是 `data`
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
    

    #torch.save(vae.state_dict(), os.path.join(output_dir, "vae.pth"))
    torch.save({
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,  # 加入 loss_history
        "learning_rate_history": lr_history  # 加入學習率歷史
    }, os.path.join(output_dir, "vae.pth"))

    print("模型與 loss 已儲存到 vae.pth")
    # 繪製 Loss 圖
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.show()
    
    # 繪製 Learning Rate 圖
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), lr_history, label="Learning Rate", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "lr_curve.png"))
    plt.show()

# **主程式**
if __name__ == "__main__":
    # 設定訓練與測試資料夾
    # 設定訓練資料夾與測試資料夾路

    #train_data_path = "Tranning data" 
    train_data_path = "/home/emily/DCAE for SIS/Pic/Baboon/Tranning data"
    #test_data_path = "Testing data"    
    test_data_path = "/home/emily/DCAE for SIS/Pic/Baboon/Testing data"

    # 產生訓練與測試影像清單
    train_image_paths = [os.path.join(train_data_path, f"baboon_{i}.png") for i in range(1,10001)]
    test_image_paths = [os.path.join(test_data_path, f"baboon_{i}.png") for i in range(1,2501)]

    # 訓練資料集與 DataLoader
    train_dataset = CustomDataset(train_image_paths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化 VAE
    #vae = VAE()
    vae = VAE().to(device)  # 讓模型搬到 GPU
    #optimizer = optim.Adam(vae.parameters(), lr=0.0022111, weight_decay=1e-5)
    #optimizer = optim.Adam(vae.parameters(), lr=0.0005, weight_decay=1e-5) #***
    optimizer = optim.Adam(vae.parameters(), lr=0.001, weight_decay=1e-5) 
    #optimizer = optim.Adam(vae.parameters(), lr=0.0005, weight_decay=1e-5)


    # 設置學習率調度器 (每 10 epochs，學習率減半)
    #scheduler = StepLR(optimizer, step_size=10, gamma=1)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    #scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    #設置學習率調度器 (每 10 epochs，學習率減半)
    #scheduler = StepLR(optimizer, step_size=10, gamma=1)
    #scheduler = StepLR(optimizer, step_size=20, gamma=0.25)
    #scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)


    # 訓練 VAE
    train_vae(vae, optimizer, scheduler,train_loader, epochs)
    #train_vae(vae, optimizer,train_loader, epochs)

    # 測試影像重建
    #test_image = preprocess_image(test_image_paths[0])
    test_image = preprocess_image(test_image_paths[0]).to(device)  # 讓測試影像也在 GPU

    with torch.no_grad():
        mu, log_var, latent, reconstructed = vae(test_image)
        sample_latent = torch.randn(latent_dim)
        n, r = 5, 3
        # 生成 shares
        shares_with_positions = create_shares(sample_latent, n, r)
        
        combined_latent = combine_shares(shares_with_positions, r).unsqueeze(0).to(device)
        

        reconstructed_from_combined = vae.decoder(combined_latent.to(device))  # 確保 latent vector 也在 GPU
        print(f"原始 latent: {sample_latent[:5]}")
        print(f"重建的 latent: {combined_latent[:5]}")
    # 顯示影像
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.view(256, 256).cpu().numpy(), cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_from_combined.view(256, 256).cpu().numpy(), cmap="gray")
    plt.title("Reconstructed from Shares")

    plt.show()

    # 儲存重建影像
    cv2.imwrite(os.path.join(output_dir, "reconstructed_image.png"), reconstructed_from_combined.view(256, 256).cpu().numpy() * 255)
    #print(next(vae.parameters()).device)  # 應該顯示 "cuda:0"
    #print(f"combined_latent device: {combined_latent.device}")  # 應該顯示 "cuda:0"
