import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def print_statistics(mse, psnr, ssim_value):
    """ 輸出 MSE, PSNR 和 SSIM 統計數據 """
    print('MSE: {:.6f}, \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(mse, psnr, ssim_value))

def calculate_mse(img1, img2):
    """ 計算 MSE（均方誤差） """
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    """ 計算 PSNR（峰值信噪比） """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """ 計算 SSIM（結構相似度） """
    return ssim(img1, img2, data_range=255, multichannel=True)

def bgr2ycbcr(img):
    """ 轉換 BGR 影像至 YCbCr 格式，並回傳 Y 通道 """
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

def plot_comparison(img1, img2, mse, psnr, ssim, output_dir):
    """ 顯示原始影像與重建影像，並標示 PSNR、SSIM 和 MSE """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(np.clip(img1 * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(np.clip(img2 * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(f"Reconstructed Image\nMSE: {mse:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    plt.axis("off")

    save_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(save_path)
    print(f"比較圖已儲存：{save_path}")
    plt.show()

def save_mse_plot(mse, output_dir):
    """ 儲存 MSE 數線圖 """
    plt.figure(figsize=(8, 4))
    plt.plot([0, 1], [mse, mse], marker='o', linestyle='-', color='red', label=f"MSE: {mse:.4f}")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, max(mse * 1.2, 50))  # MSE 數值通常較大，這樣能動態適應範圍
    plt.xticks([])
    plt.ylabel("MSE")
    plt.title("MSE Analysis")
    plt.legend()

    save_path = os.path.join(output_dir, "mse_analysis.png")
    plt.savefig(save_path)
    print(f"MSE 數線圖已儲存：{save_path}")
    plt.show()

def save_psnr_plot(psnr, output_dir):
    """ 儲存 PSNR 數線圖 """
    plt.figure(figsize=(8, 4))
    plt.plot([0, 1], [psnr, psnr], marker='o', linestyle='-', color='blue', label=f"PSNR: {psnr:.2f} dB")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 50)  # PSNR 通常範圍在 0~50 dB
    plt.xticks([])
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Analysis")
    plt.legend()

    save_path = os.path.join(output_dir, "psnr_analysis.png")
    plt.savefig(save_path)
    print(f"PSNR 數線圖已儲存：{save_path}")
    plt.show()

def save_ssim_plot(ssim_value, output_dir):
    """ 儲存 SSIM 數線圖 """
    plt.figure(figsize=(8, 4))
    plt.plot([0, 1], [ssim_value, ssim_value], marker='o', linestyle='-', color='green', label=f"SSIM: {ssim_value:.4f}")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1)  # SSIM 值範圍為 0 ~ 1
    plt.xticks([])
    plt.ylabel("SSIM")
    plt.title("SSIM Analysis")
    plt.legend()

    save_path = os.path.join(output_dir, "ssim_analysis.png")
    plt.savefig(save_path)
    print(f"SSIM 數線圖已儲存：{save_path}")
    plt.show()

def loss_curve(epochs, loss_history, output_dir):
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
    
def learning_rate(epochs, lr_history, output_dir):
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
    
def show_image(original_image, reconstructed_from_combined):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.view(256, 256).cpu().numpy(), cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_from_combined.view(256, 256).cpu().numpy(), cmap="gray")
    plt.title("Reconstructed from Shares")

    plt.show()