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
    
    util.create_directory(config['logging_params']['base_dir'])
    base_log_dir = config['logging_params']['base_dir']
    log_dir = util.get_next_version_dir(base_log_dir)
    
    util.create_directory(util.join_paths(log_dir, config['logging_params']['graph_subdir']))
    util.create_directory(util.join_paths(log_dir, config['logging_params']['recon_subdir']))
    util.create_directory(util.join_paths(log_dir, config['logging_params']['share_subdir']))
    util.save_config(config, 
                     save_path=util.join_paths(log_dir, "config_used.yml"))
    
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
    
    history = expiriment.train(train_dataloader=data.train_dataloader(),
                    # val_dataloader=data.val_dataloader(),
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    device=device)
    
    plot_graphs.loss_curve(epochs=config['exp_params']['max_epochs'],
                           loss_history=history['train_loss'],
                           output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
                           )
    
    plot_graphs.learning_rate(epochs=config['exp_params']['max_epochs'],
                              lr_history=history['learning_rate'],
                              output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
                              )
    
    print("Starting testing and secret sharing...")
    # val_dataset = data.val_dataset 
    # ran_num = torch.randint(0, len(val_dataset), (1,)).item()
    # test_image = val_dataset[ran_num].unsqueeze(0).to(device)
    
    test_image = util.get_test_image(
                        directory=config['data_params']['data_path'],
                        param=config['model_params'],
                        device=device
                        )
    
    util.save_image(test_image.squeeze(0),
                    util.join_paths(log_dir, config['logging_params']['recon_subdir'], 
                                    "original_image.png") 
                    )
    
    with torch.no_grad():
        reconstructed_image = model.generate(test_image)
        
        mu, log_var = model.encode(reconstructed_image)
        latent = model.reparameterize(mu, log_var)
        
        shares_with_positions = sss.create_shares(
                    latent.squeeze(0).cpu(), # [latent_dim]
                    n=config['shamir']['num_shares'], 
                    r=config['shamir']['threshold'],
                    output_dir=util.join_paths(log_dir, config['logging_params']['share_subdir'])
                )
        
        combined_latent = sss.combine_shares(shares_with_positions, config['shamir']['threshold']).unsqueeze(0).to(device)
        
        reconstructed_from_combined = model.decode(combined_latent)
        
        print(f"Original latent (first 5): {latent.squeeze(0)[:5]}")
        print(f"Reconstructed latent (first 5): {combined_latent.squeeze(0)[:5]}")
        
        util.save_image(reconstructed_from_combined.squeeze(0), 
                    util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_from_combined.png")
                    )
    
    
    plot_graphs.statistics(img_path1=util.join_paths(log_dir, 
                                                    config['logging_params']['recon_subdir'], 
                                                    "original_image.png"),
                            img_path2=util.join_paths(log_dir, 
                                                    config['logging_params']['recon_subdir'], 
                                                    "reconstructed_from_combined.png"), 
                            output_dir=util.join_paths(log_dir, 
                                                    config['logging_params']['graph_subdir'])
                            )
    
if __name__ == "__main__":
    main()