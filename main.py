
import os
import glob

import torch.cuda as t_cuda
import torch.optim as t_optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import device as t_device
from torch import no_grad as t_no_grad
from torch import randn as t_randn
from cv2 import imwrite as write

import vae as vae_module

import dataset as ds
import utility as util
import shamir as sss
import graph as plot_graphs

####### Configuration #######
# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

results_path = "results/"
data_path = "data/images/"

# Parameters
latent_dim = 1024  # latent dimension
epochs = 1
batch_size = 32                                                         
image_size = 256 * 256  

device = t_device("cuda" if t_cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 應該顯示 "cuda"

def main():
    start = t_cuda.Event(enable_timing=True)
    end = t_cuda.Event(enable_timing=True)
    
    image_name = "baboon"

    train_data_path = os.path.join(BASE_DIR, "Pic", image_name, "Training data")
    test_data_path  = os.path.join(BASE_DIR, "Pic", image_name, "Testing data")
    
    train_image_paths = glob.glob(os.path.join(train_data_path, "*.png"))
    test_image_paths = glob.glob(os.path.join(test_data_path, "*.png"))
    
    print(f"Found {len(train_image_paths)} training images.")

    train_dataset = ds.CustomDataset(train_image_paths)

    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    start.record()
    
    # Initialize VAE model, optimizer, and scheduler
    vae_model = vae_module.VariationalAutoencoder(image_size, latent_dim)
    vae_model = vae_model.to(device)  # 把模型搬到 GPU (Move to device)
    
    #optimizer = optim.Adam(vae.parameters(), lr=0.0022111, weight_decay=1e-5)
    #optimizer = optim.Adam(vae.parameters(), lr=0.0005, weight_decay=1e-5) #***
    optimizer = t_optim.Adam(vae_model.parameters(), lr=0.001, weight_decay=1e-5) 
    #optimizer = optim.Adam(vae.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train VAE
    
    vae_model.train_vae(optimizer, scheduler, train_loader, epochs, device)
    end.record()
    t_cuda.synchronize()
    print(f"Training time: {start.elapsed_time(end)} ms")

    # Test Image
    #test_image = preprocess_image(test_image_paths[0])
    test_image = util.preprocess_image(test_image_paths[0]).to(device)  # 讓測試影像也在 GPU
    
    with t_no_grad():
        mu, log_var, latent, reconstructed = vae_model.forward(test_image)  # 加 batch 維度
        sample_latent = t_randn(latent_dim)
        n, r = 6, 4
        # 生成 shares
        shares_with_positions = sss.create_shares(sample_latent, n, r)
        
        combined_latent = sss.combine_shares(shares_with_positions, r).unsqueeze(0).to(device)
        

        reconstructed_from_combined = vae_model.decoder(combined_latent.to(device))  # 確保 latent vector 也在 GPU
        print(f"原始 latent: {sample_latent[:5]}")
        print(f"重建的 latent: {combined_latent[:5]}")
        # 顯示影像
        plot_graphs.show_image(test_image, reconstructed_from_combined)

        # 儲存重建影像
        write(os.path.join(results_path, "reconstructed_image.png"), reconstructed_from_combined.view(256, 256).cpu().numpy() * 255)
        #print(next(vae.parameters()).device)  # 應該顯示 "cuda:0"
        #print(f"combined_latent device: {combined_latent.device}")  # 應該顯示 "cuda:0"
if __name__ == "__main__":
    main()
