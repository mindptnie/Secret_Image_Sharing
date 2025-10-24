import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# 設定資料夾
folder = "456_n13r5_output"

# 圖片路徑
original_image_path = os.path.join(folder, "grayscale_image.png")
reconstructed_image_path = os.path.join(folder, "reconstructed_image.png")

# 載入灰階圖片
original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
reconstructed_img = cv2.imread(reconstructed_image_path, cv2.IMREAD_GRAYSCALE)

# 確認圖片有正確讀到
if original_img is None:
    raise FileNotFoundError(f"找不到原圖：{original_image_path}")
if reconstructed_img is None:
    raise FileNotFoundError(f"找不到重建圖：{reconstructed_image_path}")

# 計算直方圖資料
original_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256]).flatten()
reconstructed_hist = cv2.calcHist([reconstructed_img], [0], None, [256], [0, 256]).flatten()

# === 新增區塊：計算 HDI ===
original_hist_norm = original_hist / np.sum(original_hist)
reconstructed_hist_norm = reconstructed_hist / np.sum(reconstructed_hist)
HDI = np.mean(np.abs(original_hist_norm - reconstructed_hist_norm))
print(f"Histogram Difference Index (HDI): {HDI:.6f}")

# 畫各自單獨的直方圖 —— 原圖
plt.figure(figsize=(6, 4))
plt.plot(np.arange(256), original_hist, color='blue')
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Frequency")
plt.grid(True)
save_path = os.path.join(folder, "original_histogram_single.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"原圖單獨直方圖已保存到: {save_path}")

# 畫各自單獨的直方圖 —— 重建圖
plt.figure(figsize=(6, 4))
plt.plot(np.arange(256), reconstructed_hist, color='red')
plt.title("Reconstructed Image Histogram")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Frequency")
plt.grid(True)
save_path = os.path.join(folder, "reconstructed_histogram_single.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"重建圖單獨直方圖已保存到: {save_path}")

# 畫兩張並排的比較直方圖
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(np.arange(256), original_hist, color='blue')
plt.title("(a) Original Image Histogram")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Frequency")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.arange(256), reconstructed_hist, color='red')
plt.title("(b) Reconstructed Image Histogram")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Frequency")
plt.grid(True)

save_path = os.path.join(folder, "histogram_comparison.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"並排比較版直方圖已保存到: {save_path}")
