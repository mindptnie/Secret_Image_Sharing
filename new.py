import torch
from torch import cuda
import random
import utility as util
from dataset import VAEDataModule
from model import vae_models
import shamir as sss
import graph as plot_graphs
import os
import cv2

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

def getImages(data):
    
    val_loader = data.val_dataloader()
    images = []

    for i, batch in enumerate(val_loader):
        images.append(batch)

    return images

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
        val_batch_size=1,
        img_size=params['model_params']['image_size'],
        num_channels=params['model_params']['num_channels'],
        split_ratio=0,
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory']
    )

    data.setup()
    
    model = loadModel(config['model_use']['name'],
                        param=params['model_params'],
                        path=path)

    mse_loss = 0
    psnr_value = 0
    ssim_value = 0
    i = 0
    images = getImages(data=data)

    for image in images:
        test_image = image[0].unsqueeze(0).to(device)
    
        with torch.no_grad():
            reconstructed_image = model.generate(test_image)
            
        util.save_image(
            reconstructed_image.squeeze(0), 
            util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_direct.png")
        )
        util.save_image(
            test_image.squeeze(0),
            util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png") 
        )
            
        folder_GT = util.join_paths(log_dir, config['logging_params']['recon_subdir'], "original_image.png") 
        folder_Gen = util.join_paths(log_dir, config['logging_params']['recon_subdir'], "reconstructed_direct.png")

        im_GT = cv2.imread(folder_GT) / 255.
        im_Gen = cv2.imread(folder_Gen) / 255.

        # Calculate metrics
        mse_loss += plot_graphs.calculate_mse(im_GT * 255, im_Gen * 255)
        psnr_value += plot_graphs.calculate_psnr(im_GT * 255, im_Gen * 255)
        ssim_value += plot_graphs.calculate_ssim(im_GT * 255, im_Gen * 255)
        print(f"It: {i}, MSE: {mse_loss}, PSNR: {psnr_value}, SSIM: {ssim_value}")
        i += 1
    print(f"Average MSE: {mse_loss / i}")
    print(f"Average PSNR: {psnr_value / i}")
    print(f"Average SSIM: {ssim_value / i}")


if __name__ == "__main__":
    main()