import os

import torch
from torch import cuda
from torch.backends import cudnn

from vae import VariationalAutoencoder
from dataset import VAEDataModule
from experiment import Experiment

import dataset as ds
import utility as util
import shamir as sss
import graph as plot_graphs
import warnings

warnings.filterwarnings("ignore", 
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

device = torch.device("cuda" if cuda.is_available() else "cpu")
if cuda.is_available():
    cudnn.benchmark = True 
print(f"Using device: {device}")

def main():
    
    config = util.load_config("config.yml")
    
    model = VariationalAutoencoder(image_size=config['model_params']['image_size'], 
                                  latent_dim=config['model_params']['latent_dim'],
                                  num_channels=config['model_params']['in_channels']
                                  )
    
    expiriment = Experiment(vae=model, params=config['exp_params'])
    
    data = VAEDataModule(data_path=config['data_params']['data_path'],
                        train_batch_size=config['data_params']['train_batch_size'],
                        val_batch_size=config['data_params']['val_batch_size'],
                        img_size=config['model_params']['image_size'],
                        num_channels=config['model_params']['in_channels'],
                        split_ratio=config['data_params']['split_ratio'],
                        num_workers=config['data_params']['num_workers'],
                        pin_memory=config['data_params']['pin_memory']
                        )
    data.setup()
    
    optimizer, scheduler = expiriment.configure_optimizers()
    
    expiriment.train(data.train_dataloader(), optimizer, scheduler, device)
    
    print("Loading a random test image...")
    val_dataset = data.val_dataset 
    ran_num = torch.randint(0, len(val_dataset), (1,)).item()
    test_image = val_dataset[ran_num].unsqueeze(0).to(device)
    
    util.save_image(test_image.squeeze(0), 
                os.path.join("reconstructed_outputs", "reconstructed_image.png"))
    util.create_directory("reconstructed_outputs")

    with torch.no_grad():
        reconstructed_image = model.generate(test_image)
        
        mu, log_var = model.encode(reconstructed_image)
        latent = model.reparameterize(mu, log_var)
        
        shares_with_positions = sss.create_shares(
                    latent.squeeze(0).cpu(), # [latent_dim]
                    n=config['shamir']['num_shares'], 
                    r=config['shamir']['threshold'],
                    output_dir=config['shamir']['share_path']
                )
        
        combined_latent = sss.combine_shares(shares_with_positions, config['shamir']['threshold']).unsqueeze(0).to(device)
        
        reconstructed_from_combined = model.decode(combined_latent)
        
        print(f"Original latent (first 5): {latent.squeeze(0)[:5]}")
        print(f"Reconstructed latent (first 5): {combined_latent.squeeze(0)[:5]}")
        
        util.save_image(reconstructed_from_combined.squeeze(0), 
                    os.path.join("reconstructed_outputs", "reconstructed_from_combined.png"))
    
    
    plot_graphs.calculate_statistics(img_path1=os.path.join("reconstructed_outputs", 
                                                            "reconstructed_image.png"),
                                     img_path2=os.path.join("reconstructed_outputs", 
                                                            "reconstructed_from_combined.png"), 
                                     output_dir=config['logging_params']['save_dir'],
                                     num_channels=config['model_params']['in_channels'])
    
if __name__ == "__main__":
    main()