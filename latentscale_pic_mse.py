import matplotlib.pyplot as plt
import numpy as np
import os

# 建立儲存資料夾（若尚未存在）
save_dir = "LatentPICNEW"
os.makedirs(save_dir, exist_ok=True)

# 定義資料
epochs = [1, 25, 50, 75, 100]
x_positions = np.arange(len(epochs))

# MSE 數據
mse_data = {
    'Z*0.1': [0.018, 0.00718, 0.00548, 0.00487, 0.00449],
    'Z*0.3': [0.0137, 0.00698, 0.00497, 0.00393, 0.00361],
    'Z*0.6': [0.0086, 0.00561, 0.00427, 0.00277, 0.00254],
    'Z*1.0': [0.0041, 0.00387, 0.00327, 0.00152, 0.00044],
    'Z*1.5': [0.0034, 0.00322, 0.00276, 0.00072, 0.00006]
}

# 不同 marker 樣式
markers = ['o', 's', '^', 'D', '*']  # 圓形、方形、三角形、菱形、星形

plt.figure(figsize=(8, 5))

# 繪製每條曲線，搭配不同 marker
for (label, values), marker in zip(mse_data.items(), markers):
    plt.plot(x_positions, values, marker=marker, label=label)
    for x, y in zip(x_positions, values):
        plt.text(x, y, f"{y:.5f}", fontsize=8, ha='center', va='bottom')

# 圖表設定
plt.title("MSE vs Epoch with Different Latent Scales")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.xticks(x_positions, epochs)
plt.legend(title="LatentScale")
plt.grid(True)
plt.tight_layout()

# 儲存圖片
save_path = os.path.join(save_dir, "mse_plot.png")
plt.savefig(save_path, dpi=300)
plt.show()
