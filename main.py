import math
import os
import glob

import torch.cuda as t_cuda
import torch.optim as t_optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import device as t_device
from torch import no_grad as t_no_grad
from torch import randn as t_randn
from cv2 import imwrite as write
from PIL import Image
import torchvision.transforms as transforms
import torch

import vae as vae_module

import dataset as ds
import utility as util
import shamir as sss
import graph as plot_graphs

# Root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#vvvvvv Configuration vvvvvv#

# Directories
process_path = "preprocessed"
graphs_path = "graph_outputs"
results_path = "results"
data_path = "Pic"

# Name of the image folder  
image_name = "cat_dog"

# Parameters
latent_dim = 1024  # latent dimension
epochs = 1
batch_size = 32
image_size = (128, 128)
lambda_weight = 0.0001  # weight for KL divergence loss

# Image Channels
img_channels = 1  # 1 = Grayscale images , 3 = RGB images

# Shamir's Secret Sharing parameters
n_shares = 5  # Share count
r_threshold = 3  # Reconstruction threshold

#^^^^^^ Configuration ^^^^^^#

device = t_device("cuda" if t_cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():

    # Create necessary directories
    util.create_directory(process_path)
    util.create_directory(results_path)
    util.create_directory(data_path)
    util.create_directory(graphs_path)
    
    # Delete previous shares
    util.delete_image(os.path.join(BASE_DIR, "shares"))
    
    start = t_cuda.Event(enable_timing=True)
    end = t_cuda.Event(enable_timing=True)
    
    train_data_path = os.path.join(BASE_DIR, data_path, image_name, "Training data")
    test_data_path  = os.path.join(BASE_DIR, data_path, image_name, "Testing data")
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Training or Testing data path does not exist. Please check the directory: {train_data_path} or {test_data_path}")

    train_dataset = ds.CustomDataset(root_dir=train_data_path, 
                                     target_size=image_size,
                                     num_channels=img_channels)
    
    test_dataset = ds.CustomDataset(root_dir=test_data_path, 
                                     target_size=image_size,
                                     num_channels=img_channels)

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8 
    )
    
    start.record()
    
    # Initialize VAE model 
    vae_model = vae_module.VariationalAutoencoder(image_size=image_size, 
                                                  latent_dim=latent_dim)
    vae_model = vae_model.to(device) # Move to GPU
    
    optimizer = t_optim.Adam(
        vae_model.parameters(), 
        lr=0.001, 
        weight_decay=1e-5) 
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train VAE
    vae_model.train_vae(optimizer, scheduler, train_loader, epochs, device, lambda_weight=lambda_weight)
    end.record()
    t_cuda.synchronize()
    print(f"Training time: {start.elapsed_time(end)/60000} mins")

    # Test Image
    test_image = test_dataset.__getitem__(0).to(device)  # 讓測試影像也在 GPU

    with t_no_grad():
        mu, log_var, latent, reconstructed = vae_model.forward(test_image)
        sample_latent = t_randn(latent_dim)

        # Generate shares
        shares_with_positions = sss.create_shares(
                    latent.squeeze(0).cpu(), 
                    n_shares, 
                    r_threshold
                )        
        combined_latent = sss.combine_shares(shares_with_positions, r_threshold).unsqueeze(0).to(device)

        # FIXED: Use decoder_input layer to project latent to decoder input shape
        decoder_input = vae_model.decoder_input(combined_latent)
        decoder_input = decoder_input.view(-1, 256, vae_model.final_conv_size, vae_model.final_conv_size)
        reconstructed_from_combined = vae_model.decoder(decoder_input)
        
        print(f"原始 latent: {sample_latent[:5]}")
        print(f"重建的 latent: {combined_latent[:5]}")
    
    # Save reconstructed image - it's already [1, 1, 256, 256]
    reconstructed_image = reconstructed_from_combined.squeeze().cpu().numpy() * 255
    write(os.path.join(BASE_DIR, results_path, "reconstructed_image.png"), reconstructed_image)
    
    plot_graphs.calculate_statistics(process_path, results_path)
    
    
if __name__ == "__main__":
    main()