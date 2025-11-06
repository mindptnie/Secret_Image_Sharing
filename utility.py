
import os
import cv2
import glob
import yaml
from PIL import Image
import torch.utils.data
from torchvision.transforms import ToTensor

def covert_rgba(img:Image, img_size:int, num_channels:int=1):
    if num_channels == 1:
            img = img.convert('L')
    else:
        if img.mode in ('RGBA', 'LA', 'P', 'PA'):
            background = Image.new("RGB", img.size, (255, 255, 255))
            img_rgba = img.convert('RGBA')
            background.paste(img_rgba, mask=img_rgba.split()[3]) 
            img = background
        else:
            img = img.convert('RGB')
    img = img.resize((img_size, img_size))
    
    return img

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

def load_config(config_path="config.yml"):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None