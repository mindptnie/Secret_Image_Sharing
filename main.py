import os
import torch
import random
from torch import cuda
from torch.backends import cudnn

from vae import VQVariationalAutoencoder
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
    
    # Create VQ-VAE model instead of VAE
    model = VQVariationalAutoencoder(
        image_size=config['model_params']['image_size'], 
        num_embeddings=config['model_params'].get('num_embeddings', 512),  # Codebook size
        embedding_dim=config['model_params'].get('embedding_dim', 64),      # Code dimension
        num_channels=config['model_params']['in_channels'],
        commitment_cost=config['model_params'].get('commitment_cost', 0.25)
    )
    
    print(f"VQ-VAE created with codebook size: {model.num_embeddings}, embedding dim: {model.embedding_dim}")
    
    experiment = Experiment(vae=model, params=config['exp_params'])
    
    data = VAEDataModule(
        data_path=config['data_params']['data_path'],
        train_batch_size=config['data_params']['train_batch_size'],
        val_batch_size=config['data_params']['val_batch_size'],
        img_size=config['model_params']['image_size'],
        num_channels=config['model_params']['in_channels'],
        split_ratio=config['data_params']['split_ratio'],
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory']
    )
    data.setup()
    
    optimizer, scheduler = experiment.configure_optimizers()
    
    print("Starting training...")
    history = experiment.train(
        train_dataloader=data.train_dataloader(),
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device
    )
    
    # Plot training curves
    plot_graphs.loss_curve(
        epochs=config['exp_params']['max_epochs'],
        loss_history=history['train_loss'],
        output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
    )
    
    plot_graphs.learning_rate(
        epochs=config['exp_params']['max_epochs'],
        lr_history=history['learning_rate'],
        output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
    )
    
    print("\n" + "="*50)
    print("Starting testing and secret sharing...")
    print("="*50)
    
    # Get a batch from validation set
    # val_loader = data.val_dataloader()
    # random_batch_idx = random.randint(0, len(val_loader) - 1)
    # for i, batch in enumerate(val_loader):
    #     if i == random_batch_idx:
    #         test_images = batch
    #         break
    # test_image = test_images[0].unsqueeze(0).to(device)

    test_image = util.get_test_image(
                        directory=config['data_params']['data_path'],
                        param=config['model_params'],
                        device=device
                        )    
    util.save_image(
        test_image.squeeze(0),
        util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png") 
    )
    
    with torch.no_grad():
        # Get reconstruction and codebook indices
        reconstructed_image = model.generate(test_image)
        encoding_indices = model.get_codebook_indices(test_image)
        
        print(f"\nCodebook indices shape: {encoding_indices.shape}")
        print(f"Indices range: [{encoding_indices.min().item()}, {encoding_indices.max().item()}]")
        print(f"Unique indices used: {len(torch.unique(encoding_indices))}/{model.num_embeddings}")
        
        # Create Shamir shares from codebook indices
        print(f"\nCreating {config['shamir']['num_shares']} shares with threshold {config['shamir']['threshold']}...")
        shares_with_positions = sss.create_shares_from_indices(
            encoding_indices,
            n=config['shamir']['num_shares'], 
            r=config['shamir']['threshold'],
            output_dir=util.join_paths(log_dir, config['logging_params']['share_subdir'])
        )
        
        # Reconstruct from shares
        print(f"\nRecombining shares (using threshold={config['shamir']['threshold']} shares)...")
        reconstructed_indices = sss.combine_shares_to_indices(
            shares_with_positions, 
            config['shamir']['threshold'],
            shape=(encoding_indices.shape[1], encoding_indices.shape[2])
        ).to(device)
        
        print(f"Reconstructed indices shape: {reconstructed_indices.shape}")
        
        # Check reconstruction accuracy
        indices_match = torch.equal(encoding_indices, reconstructed_indices)
        print(f"Indices perfectly reconstructed: {indices_match}")
        if not indices_match:
            diff = (encoding_indices != reconstructed_indices).sum().item()
            total = encoding_indices.numel()
            print(f"Mismatched indices: {diff}/{total} ({100*diff/total:.2f}%)")
        
        # Decode from reconstructed indices
        reconstructed_from_shares = model.decode_from_indices(reconstructed_indices)
        
        # Save reconstructed image
        util.save_image(
            reconstructed_from_shares.squeeze(0), 
            util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_from_shares.png")
        )
        
        # Also save direct reconstruction for comparison
        util.save_image(
            reconstructed_image.squeeze(0), 
            util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_direct.png")
        )
    
    # Plot statistics
    print("\nGenerating comparison statistics...")
    plot_graphs.statistics(
        img_path1=util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png"),
        img_path2=util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_from_shares.png"), 
        output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
    )
    
    print("\n" + "="*50)
    print("VQ-VAE training and testing completed!")
    print(f"Results saved to: {log_dir}")
    print("="*50)

if __name__ == "__main__":
    main()