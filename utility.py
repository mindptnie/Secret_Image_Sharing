import os
import cv2
import torch.nn as t_nn
from torch import sum as t_sum

def vae_loss_function(reconstructed, original, mu, log_var):
     # **確保 reconstructed 的 shape 和 original 一樣**
    assert reconstructed.shape == original.shape, f"Shape mismatch: {reconstructed.shape} vs {original.shape}"

    recon_loss = t_nn.MSELoss()(reconstructed, original) # GPU 計算
    kl_loss = -0.5 * t_sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL 散度  # GPU 計算
    return recon_loss + 0.0001 * kl_loss  # 調整 KL loss 權重

# 預處理影像
def preprocess_image(image_path, output_dir="preprocessed", image_size=256*256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join(output_dir, "grayscale_image.png"), img_resized)
    return t_nn.ToTensor()(img_resized).view(-1, image_size).float()

