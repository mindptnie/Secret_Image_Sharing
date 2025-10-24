import matplotlib.pyplot as plt
import numpy as np
import os

# 建立儲存資料夾（若尚未存在）
save_dir = "LatentPICNEW"
os.makedirs(save_dir, exist_ok=True)

# 定義資料
epochs = [1, 25, 50, 75, 100]
x_positions = np.arange(len(epochs))

# PSNR 數據
psnr_data = {
    'Z*0.1': [17.44, 21.43, 22.60, 23.12, 23.46],
    'Z*0.3': [18.63, 21.56, 23.02, 24.05, 24.42],
    'Z*0.6': [20.64, 22.50, 23.68, 25.57, 25.94],
    'Z*1.0': [23.91, 24.12, 24.84, 28.16, 33.47],
    'Z*1.5': [24.71, 24.91, 25.57, 31.39, 42.07]
}

# 不同 marker 樣式
markers = ['o', 's', '^', 'D', '*']  # 圓形、方形、三角形、菱形、星形

# 開始繪圖
plt.figure(figsize=(8, 5))

# 繪製每條曲線，搭配不同 marker
for (label, values), marker in zip(psnr_data.items(), markers):
    plt.plot(x_positions, values, marker=marker, label=label)
    for x, y in zip(x_positions, values):
        plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')

# 圖表設定
plt.title("PSNR vs Epoch with Different Latent Scales")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.xticks(x_positions, epochs)
plt.legend(title="LatentScale")
plt.grid(True)
plt.tight_layout()

# 儲存圖片
save_path = os.path.join(save_dir, "psnr_plot.png")
plt.savefig(save_path, dpi=300)
plt.show()
