import os
import torch
import cv2

import torch.cuda as t_cuda
import torch.optim as t_optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import device as t_device
from torch import no_grad as t_no_grad
from torch import randn as t_randn
import torch.backends.cudnn as cudnn

from vae import VariationalAutoencoder
from experiment import VAEXperiment
from dataset import VAEDataModule
import utility

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import seed_everything


import dataset as ds
import shamir as sss
import graph as plot_graphs

# Root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# device = t_device("cuda" if t_cuda.is_available() else "cpu")
# if t_cuda.is_available():
#     cudnn.benchmark = True 
# print(f"Using device: {device}")

def main():
    
    config = utility.load_config("config.yml")
    
    seed_everything(config['model_params']['manual_seed'], True)
    
    model = VariationalAutoencoder(config['model_params']['image_size'],
                                       config['model_params']['latent_dim'],
                                       config['model_params']['in_channels'])
    
    data = VAEDataModule(data_path=config['data_params']['data_path'],
                        train_batch_size=config['data_params']['train_batch_size'],
                        val_batch_size=config['data_params']['val_batch_size'],
                        patch_size=config['model_params']['image_size'],
                        num_channels=config['model_params']['in_channels'],
                        num_workers=config['data_params']['num_workers'],
                        pin_memory=config['data_params']['pin_memory']
                        )

    
    experiment = VAEXperiment(vae_model=model,
                              params=config['exp_params'])    
    
    bar_callback = RichProgressBar(refresh_rate=1)

    trainer = Trainer(
        max_epochs=config['trainer_params']['max_epochs'],
        
        callbacks=[bar_callback] 
    )
    
    trainer.fit(experiment,data)
    # # Create necessary directories
    # util.create_directory(process_path)
    # util.create_directory(results_path)
    # util.create_directory(data_path)
    # util.create_directory(graphs_path)
    
    # # Delete previous shares
    # util.delete_image(os.path.join(BASE_DIR, "shares"))
    
    # start = t_cuda.Event(enable_timing=True)
    # end = t_cuda.Event(enable_timing=True)
    
    # train_data_path = os.path.join(BASE_DIR, data_path, image_name, "Training data")
    # test_data_path  = os.path.join(BASE_DIR, data_path, image_name, "Testing data")
    
    # if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
    #     raise FileNotFoundError(f"Training or Testing data path does not exist. Please check the directory: {train_data_path} or {test_data_path}")

    # train_dataset = ds.CustomDataset(root_dir=train_data_path, 
    #                                  target_size=image_size,
    #                                  in_channels=img_channels)
    
    # test_dataset = ds.CustomDataset(root_dir=test_data_path, 
    #                                  target_size=image_size,
    #                                  in_channels=img_channels)

    # train_loader = train_dataset.get_dataloader(
    #     batch_size=batch_size, 
    #     shuffle=True, 
    #     num_workers=8 
    # )
    
    # start.record()
    
    # # Initialize VAE model 
    # vae_model = vae_module.VariationalAutoencoder(image_size=image_size, 
    #                                               latent_dim=latent_dim,
    #                                               in_channels=img_channels)
    # vae_model = vae_model.to(device) # Move to GPU
    
    # optimizer = t_optim.AdamW(
    #     vae_model.parameters(), 
    #     lr=0.001, 
    #     weight_decay=1e-5) 
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # # Train VAE
    # vae_model.train_vae(optimizer, scheduler, train_loader, epochs, device, lambda_weight=lambda_weight)
    # end.record()
    # t_cuda.synchronize()
    # print(f"Training time: {start.elapsed_time(end)/60000} mins")

    # # Test Image
    # ran_num = torch.randint(0, len(test_dataset), (1,)).item()
    # test_image = test_dataset.__getitem__(ran_num).unsqueeze(0).to(device)  # 讓測試影像也在 GPU

    # with t_no_grad():
    #     mu, log_var, latent, reconstructed = vae_model.forward(test_image)
    #     sample_latent = t_randn(latent_dim)

    #     # Generate shares
    #     shares_with_positions = sss.create_shares(
    #                 latent.squeeze(0).cpu(), 
    #                 n_shares, 
    #                 r_threshold
    #             )        
    #     combined_latent = sss.combine_shares(shares_with_positions, r_threshold).unsqueeze(0).to(device)

    #     # FIXED: Use decoder_input layer to project latent to decoder input shape
    #     decoder_input = vae_model.decoder_input(combined_latent)
    #     decoder_input = decoder_input.view(-1, 256, vae_model.final_conv_size, vae_model.final_conv_size)
    #     reconstructed_from_combined = vae_model.decoder(decoder_input)
        
    #     print(f"原始 latent: {sample_latent[:5]}")
    #     print(f"重建的 latent: {combined_latent[:5]}")
    
    # save_path = os.path.join(BASE_DIR, results_path, "reconstructed_image.png")
    
    # if img_channels == 1:
    #     # Grayscale
    #     reconstructed_image = reconstructed_from_combined.squeeze().cpu().numpy() * 255
    #     cv2.imwrite(save_path, reconstructed_image)
    # else:
    #     # RGB
    #     reconstructed_img = reconstructed_from_combined.squeeze(0).cpu() # (3, 128, 128)
    #     reconstructed_img = reconstructed_img.permute(1, 2, 0).numpy() * 255 # (128, 128, 3)
        
    #     # แปลงจาก RGB (PyTorch) เป็น BGR (OpenCV)
    #     reconstructed_bgr = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(save_path, reconstructed_bgr)
    
    # print(f"Reconstructed image saved to {save_path}")
    
    # plot_graphs.calculate_statistics(process_path, results_path, in_channels=img_channels)
    
    
if __name__ == "__main__":
    main()