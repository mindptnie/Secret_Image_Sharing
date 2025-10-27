import math
import os
import cv2
import torch.nn as t_nn
from torch import sum as t_sum
import torch.utils.data
from torchvision.transforms import ToTensor

def vae_loss_function(reconstructed, original, mu, log_var):
     # **確保 reconstructed 的 shape 和 original 一樣**
    assert reconstructed.shape == original.shape, f"Shape mismatch: {reconstructed.shape} vs {original.shape}"

    recon_loss = t_nn.MSELoss()(reconstructed, original) # GPU 計算
    kl_loss = -0.5 * t_sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL 散度  # GPU 計算
    return recon_loss + 0.0001 * kl_loss  # 調整 KL loss 權重

# 預處理影像
def preprocess_image(image_path, image_size=256*256, output_dir="preprocessed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    size = int(math.sqrt(image_size))

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    img_resized = cv2.resize(img, (size, size))
    cv2.imwrite(os.path.join(output_dir, "grayscale_image.png"), img_resized)
    return ToTensor()(img_resized).view(-1, image_size).float()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)