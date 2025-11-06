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

import warnings 


warnings.filterwarnings("ignore", 
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")
def main():
    
    config = utility.load_config("config.yml")
    
    seed_everything(config['model_params']['manual_seed'], True)
    
    model = VariationalAutoencoder(config['model_params']['image_size'],
                                       config['model_params']['latent_dim'],
                                       config['model_params']['in_channels'])
    
    data = VAEDataModule(data_path=config['data_params']['data_path'],
                        train_batch_size=config['data_params']['train_batch_size'],
                        val_batch_size=config['data_params']['val_batch_size'],
                        img_size=config['model_params']['image_size'],
                        num_channels=config['model_params']['in_channels'],
                        num_workers=config['data_params']['num_workers'],
                        pin_memory=config['data_params']['pin_memory']
                        )

    
    experiment = VAEXperiment(vae_model=model,
                              params=config['exp_params'])    
    
    bar_callback = RichProgressBar(refresh_rate=1)

    trainer = Trainer(callbacks=[bar_callback],
                      **config['trainer_params']
    )
    
    trainer.fit(experiment,data)
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