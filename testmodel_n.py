import torch
from torch import cuda
import random
import os
import utility as util
from dataset import VAEDataModule
from model import vae_models
import shamir as sss

device = torch.device("cuda" if cuda.is_available() else "cpu")

#### Configuration
VERSION = "0" 
START_INDEX = 1    
NUM_IMAGES = 5     
EPOCH = 1

def loadModel(name:str,param,path:str):
    model_type = name
    model = vae_models[model_type](**param)
    model = torch.load(path, map_location=device)
    print(f"Model loaded from {path}")
    return model

def loadImageIndex(data, index:int = 0):
    val_loader = data.val_dataloader()
    test_images = None
    for i, batch in enumerate(val_loader):
        if i == index:
            test_images = batch
            break 
    if test_images is None:
        return loadImage(data)
    test_image = test_images[0].unsqueeze(0).to(device)
    return test_image

def loadImage(data):
    val_loader = data.val_dataloader()
    test_images = next(iter(val_loader))
    test_image = test_images[0].unsqueeze(0).to(device)
    return test_image

def main():
    config = util.load_config("config.yml")
    log_dir = config['logging_params']['base_dir']+f"version_{VERSION}/"
    config = util.load_config(f"{log_dir}config_used.yml")
    model_name = config['model_use']['name']
    path = f"{log_dir}{model_name}_model_epoch_{EPOCH}.pth"
    params = util.load_config(f"{log_dir}{model_name}.yml")
    
    util.create_directory(util.join_paths(log_dir, config['logging_params']['recon_subdir']))
    util.create_directory(util.join_paths(log_dir, config['logging_params']['share_subdir']))
    
    data = VAEDataModule(
        data_path=config['data_params']['data_path'],
        train_batch_size=config['data_params']['train_batch_size'],
        val_batch_size=config['data_params']['val_batch_size'],
        img_size=params['image_size'],
        num_channels=params['num_channels'],
        split_ratio=config['data_params']['split_ratio'],
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory']
    )
    data.setup()

    model = loadModel(config['model_use']['name'], param=params, path=path)

    original_imgs_list = []
    recon_imgs_list = []

    print(f"Starting processing {NUM_IMAGES} images...")

    for i in range(NUM_IMAGES):
        current_idx = START_INDEX + i
        
        # 1. Load Image
        test_image = loadImageIndex(data=data, index=current_idx)
        
        with torch.no_grad():
            # 2. Reconstruction Process
            if hasattr(model, 'get_codebook_indices'):
                # VQ-VAE Logic
                encoding_indices = model.get_codebook_indices(test_image)
                
                temp_share_dir = util.join_paths(log_dir, config['logging_params']['share_subdir'], "temp_processing")
                if not os.path.exists(temp_share_dir): os.makedirs(temp_share_dir, exist_ok=True)

                shares_with_positions = sss.create_shares_from_indices(
                    encoding_indices,
                    n=config['shamir']['num_shares'], 
                    r=config['shamir']['threshold'],
                    codebook_size=model.codebook_size,
                    output_dir=temp_share_dir
                )
                
                reconstructed_indices = sss.combine_shares_to_indices(
                    shares_with_positions, 
                    config['shamir']['threshold'],
                    codebook_size=model.codebook_size,
                    shape=(encoding_indices.shape[1], encoding_indices.shape[2])
                ).to(device)
                
                reconstructed_from_shares = model.decode_from_indices(reconstructed_indices)
                
            else:
                # VAE Logic
                mu, log_var = model.encode(test_image)
                temp_share_dir = util.join_paths(log_dir, config['logging_params']['share_subdir'], "temp_processing_legacy")
                if not os.path.exists(temp_share_dir): os.makedirs(temp_share_dir, exist_ok=True)

                shares_with_positions = sss.create_shares_legacy(mu.view(-1), config['shamir']['num_shares'], config['shamir']['threshold'], temp_share_dir)
                reconstructed_latent = sss.combine_shares_legacy(shares_with_positions, config['shamir']['threshold']).to(device).unsqueeze(0)
                reconstructed_from_shares = model.decode(reconstructed_latent)

        original_imgs_list.append(test_image)
        recon_imgs_list.append(reconstructed_from_shares)
        
        print(f"Collected Image {i+1}/{NUM_IMAGES}")

    print("Constructing image rows...")

    row_original = torch.cat(original_imgs_list, dim=3)
    
    path_original = util.join_paths(log_dir, config['logging_params']['recon_subdir'], f"row_original_start{START_INDEX}_n{NUM_IMAGES}.png")
    util.save_image(row_original.squeeze(0), path_original)
    print(f"Saved Original Row to: {path_original}")

    row_recon = torch.cat(recon_imgs_list, dim=3)
    
    path_recon = util.join_paths(log_dir, config['logging_params']['recon_subdir'], f"row_recon_start{START_INDEX}_n{NUM_IMAGES}_c{params['codebook_size']}*{params['codebook_dim']}.png")
    util.save_image(row_recon.squeeze(0), path_recon)
    print(f"Saved Recon Row to: {path_recon}")

    print("Done.")

if __name__ == "__main__":
    main()