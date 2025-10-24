import matplotlib.pyplot as plt
import numpy as np
import os

# 建立儲存資料夾（若尚未存在）
save_dir = "LatentPICNEW"
os.makedirs(save_dir, exist_ok=True)

# 定義資料
epochs = [1, 25, 50, 75, 100]
x_positions = np.arange(len(epochs))

# SSIM 數據
ssim_data = {
    'Z*0.1': [0.3079, 0.7315, 0.8103, 0.8264, 0.8408],
    'Z*0.3': [0.4708, 0.7399, 0.8313, 0.8639, 0.8755],
    'Z*0.6': [0.6735, 0.7971, 0.8609, 0.9075, 0.9153],
    'Z*1.0': [0.8677, 0.8800, 0.9072, 0.9500, 0.9837],
    'Z*1.5': [0.9012, 0.9314, 0.9365, 0.9872, 0.9988]
}

# 不同 marker 樣式
markers = ['o', 's', '^', 'D', '*']  # 圓形、方形、三角形、菱形、星形

# 開始繪圖
plt.figure(figsize=(8, 5))

# 繪製每條曲線，搭配不同 marker
for (label, values), marker in zip(ssim_data.items(), markers):
    plt.plot(x_positions, values, marker=marker, label=label)
    for x, y in zip(x_positions, values):
        plt.text(x, y, f"{y:.4f}", fontsize=8, ha='center', va='bottom')

# 圖表設定
plt.title("SSIM vs Epoch with Different Latent Scales")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.xticks(x_positions, epochs)
plt.legend(title="LatentScale")
plt.grid(True)
plt.tight_layout()

# 儲存圖片
save_path = os.path.join(save_dir, "ssim_plot.png")
plt.savefig(save_path, dpi=300)
plt.show()
