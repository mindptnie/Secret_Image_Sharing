import math
import os
import cv2
import glob
from PIL import Image

import torch.nn as t_nn
from torch import sum as t_sum
from torch import mean as t_mean
import torch.utils.data
from torchvision.transforms import ToTensor

# 預處理影像
def preprocess_image(image:Image, image_size=(256,256), num_channels:int=1, output_dir="preprocessed"):
    create_directory(output_dir)

    if num_channels == 1:
        image = image.convert('L')
    else:
        if image.mode in ('RGBA', 'LA', 'P', 'PA'):

            background = Image.new("RGB", image.size, (255, 255, 255))
            
            image_rgba = image.convert('RGBA')
            
            background.paste(image_rgba, mask=image_rgba.split()[3]) 
            image = background
            
        else:
            image = image.convert('RGB')
            
    img_resized = image.resize((image_size[0], image_size[1]))
    # img_resized.save(os.path.join(output_dir, save_name))
    
    return ToTensor()(img_resized).float()

def save_image(tensor_image, save_path:str):
    tensor_image = tensor_image.clamp(0, 1)
    pil_image = ToTensor().to_pil_image(tensor_image.cpu())
    pil_image.save(save_path)

def load_image(image_path:str):
    image = Image.open(image_path)
    return image

def create_directory(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
        
def delete_image(image_path:str):
    for img in glob.glob(os.path.join(image_path, f"*.png")):
        os.remove(img)

def get_images_in_paths(folder_path:str,img_name="*.png"):
    if img_name!="*.png":
        img_name = f"{img_name}*.png"
    return glob.glob(os.path.join(folder_path, img_name))