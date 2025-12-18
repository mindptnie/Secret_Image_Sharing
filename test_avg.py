import torch
from torch import cuda
import random
import utility as util
from dataset import VAEDataModule
from model import vae_models
import shamir as sss
import graph as plot_graphs


device = torch.device("cuda" if cuda.is_available() else "cpu")


#### Configuration
VERSION = "98" 
image_index = 5

def loadModel(name:str,param,path:str):

    model_type = name
    model = vae_models[model_type](**param)
    
    model = torch.load(path, map_location=device)
    # model.load_state_dict(torch.load(path, map_location=device))

    print("model is loaded")
    return model

def loadImage(data):
    
    val_loader = data.val_dataloader()
    random_batch_idx = random.randint(0, len(val_loader) - 1)

    for i, batch in enumerate(val_loader):
        if i == random_batch_idx:
            test_images = batch
            break 

    test_image = test_images[0].unsqueeze(0).to(device)

    return test_image

def loadImageIndex(data,index:int = 0):
    val_loader = data.val_dataloader()

    for i, batch in enumerate(val_loader):
        if i == index:
            test_images = batch
            break 

    test_image = test_images[0].unsqueeze(0).to(device)

    return test_image

def main():
    config = util.load_config("config.yml")
    log_dir = config['logging_params']['base_dir']+"version_"+VERSION+"/"
    config = util.load_config(log_dir+"config_used.yml")
    model_name = config['model_use']['name']
    path = log_dir+model_name+"_model.pth"
    params = util.load_config(log_dir+model_name+".yml")
    data = VAEDataModule(
        data_path=config['data_params']['data_path'],
        train_batch_size=config['data_params']['train_batch_size'],
        val_batch_size=config['data_params']['val_batch_size'],
        img_size=params['model_params']['image_size'],
        num_channels=params['model_params']['num_channels'],
        split_ratio=config['data_params']['split_ratio'],
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory']
    )

    data.setup()

    model = loadModel(config['model_use']['name'],
                        param=params['model_params'],
                        path=path)
    # Statistics accumulators
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0
    
    val_loader = data.val_dataloader()
    print(f"Starting evaluation on validation set ({len(val_loader)} batches)...")
    
    # Iterate over all batches in validation loader
    for batch_idx, batch in enumerate(val_loader):
        # batch is a tensor of images
        images = batch.to(device)
        
        for i in range(images.size(0)):
            test_image = images[i].unsqueeze(0) # [1, C, H, W]
            
            with torch.no_grad():
                # Get reconstruction
                reconstructed_image = model.generate(test_image)
                
                if hasattr(model, 'get_codebook_indices'):
                    # VQ-VAE Logic
                    encoding_indices = model.get_codebook_indices(test_image)
                    
                    # Create Shamir shares from codebook indices
                    shares_with_positions = sss.create_shares_from_indices(
                        encoding_indices,
                        n=config['shamir']['num_shares'], 
                        r=config['shamir']['threshold'],
                        output_dir=None # Do not save every share to disk to save space/time
                    )
                    
                    # Reconstruct from shares
                    reconstructed_indices = sss.combine_shares_to_indices(
                        shares_with_positions, 
                        config['shamir']['threshold'],
                        shape=(encoding_indices.shape[1], encoding_indices.shape[2])
                    ).to(device)
                    
                    # Decode from reconstructed indices
                    reconstructed_from_shares = model.decode_from_indices(reconstructed_indices)
                    
                else:
                    # VAE Logic (Legacy/Continuous)
                    mu, log_var = model.encode(test_image)
                    z_flat = mu.view(-1)
                    
                    shares_with_positions = sss.create_shares_legacy(
                        z_flat,
                        n=config['shamir']['num_shares'], 
                        r=config['shamir']['threshold'],
                        output_dir=None
                    )
                    
                    reconstructed_latent = sss.combine_shares_legacy(
                        shares_with_positions, 
                        config['shamir']['threshold']
                    ).to(device)
                    
                    reconstructed_latent = reconstructed_latent.unsqueeze(0)
                    reconstructed_from_shares = model.decode(reconstructed_latent)

            # Calculate metrics for this image
            # Convert tensors to numpy [H, W, C] in 0-255 range
            img_gt = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            img_recon = reconstructed_from_shares.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            
            # Clip values to ensure valid range
            img_gt = img_gt.clip(0, 255)
            img_recon = img_recon.clip(0, 255)
            
            mse = plot_graphs.calculate_mse(img_gt, img_recon)
            psnr = plot_graphs.calculate_psnr(img_gt, img_recon)
            ssim_val = plot_graphs.calculate_ssim(img_gt, img_recon)
            
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim_val
            total_images += 1
            
            if total_images % 10 == 0:
                print(f"Processed {total_images} images...")

    # Calculate averages
    avg_mse = total_mse / total_images
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    
    print("\n" + "="*30)
    print(f"Evaluation Complete")
    print(f"Total Images: {total_images}")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.6f} dB")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print("="*30 + "\n")

    # Save the last processed image as a sample
    print("Saving last processed image as sample...")
    util.save_image(
        test_image.squeeze(0),
        util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png") 
    )
    util.save_image(
        reconstructed_from_shares.squeeze(0), 
        util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_from_shares.png")
    )
    
    # Plot statistics for the last sample
    plot_graphs.statistics(
        img_path1=util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png"),
        img_path2=util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_from_shares.png"), 
        output_dir=util.join_paths(log_dir, config['logging_params']['graph_subdir'])
    )

    # Plot average statistics
    print("\nGenerating average statistics graphs...")
    graph_avg_dir = util.join_paths(log_dir, "graph_avg")
    util.create_directory(graph_avg_dir)
    
    # Use the relative path for plot_graphs because it joins with BASE_DIR internally
    # We need to be careful. Let's look at graph.py.
    # graph.py: folder_GT = os.path.join(BASE_DIR, img_path1)
    # The output_dir argument in save_mse_plot is used as: os.path.join(BASE_DIR, output_dir, "mse_analysis.png")
    # So we should pass a relative path from BASE_DIR (which is where graph.py is).
    # log_dir starts with 'logs/...', which is relative to where the script is run (usually).
    # But wait, BASE_DIR in graph.py is os.path.dirname(os.path.abspath(__file__)).
    # If graph.py is in /.../Secret_Image_Sharing/, then BASE_DIR is that folder.
    # log_dir is constructed as config['logging_params']['base_dir'] + ...
    # config['logging_params']['base_dir'] is usually "logs/".
    
    # We can pass the relative path "logs/version_XX/graph_avg"
    
    relative_avg_dir = util.join_paths("logs", f"version_{VERSION}", "graph_avg")
    
    plot_graphs.save_mse_plot(avg_mse, relative_avg_dir)
    plot_graphs.save_psnr_plot(avg_psnr, relative_avg_dir)
    plot_graphs.save_ssim_plot(avg_ssim, relative_avg_dir)

if __name__ == "__main__":
    main()