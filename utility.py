
import os
import glob
import yaml
import re
import shutil
from PIL import Image
import torch.utils.data
from torchvision.transforms import ToPILImage,ToTensor

def covert_to_png(image_path:str, save_path:str):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img.save(save_path, 'PNG')

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

def covert_to_tensor(img:Image, img_size:int, num_channels:int=1):
    img = covert_rgba(img, img_size, num_channels)
    tensor_img = ToTensor()(img)
    return tensor_img

def save_image(tensor_image, save_path:str):
    tensor_image = tensor_image.clamp(0, 1)
    pil_image = ToPILImage()(tensor_image.cpu())
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

def get_img_files_in_directory(directory:str, pattern:str="*.png")->list:
    search_pattern = os.path.join(directory, pattern)
    return glob.glob(search_pattern)

def load_config(config_path="config.yml"):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

def save_config(config:dict, save_path="config_saved.yml"):
    with open(save_path, 'w') as file:
        try:
            yaml.dump(config, file)
            print(f"Configuration saved to: {save_path}")
        except yaml.YAMLError as exc:
            print(f"Error saving YAML file: {exc}")

def get_next_version_dir(base_log_dir="logs")->str:
    os.makedirs(base_log_dir, exist_ok=True)
    
    existing_dirs = [d for d in os.listdir(base_log_dir) 
                     if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")]
    
    next_version = 0
    if existing_dirs:
        versions = [int(re.search(r'version_(\d+)', d).group(1)) 
                    for d in existing_dirs if re.search(r'version_(\d+)', d)]
        if versions:
            next_version = max(versions) + 1
            
    log_dir = os.path.join(base_log_dir, f"version_{next_version}")
    os.makedirs(log_dir)
    
    print(f"Logs and outputs will be saved to: {log_dir}")

    return log_dir

def join_paths(*paths)->str:
    return os.path.join(*paths)

def get_test_image(directory:str, param:dict,device)->torch.Tensor:
    test_image_path = get_img_files_in_directory(
                            directory=directory,
                            pattern="1 (1644).jpg"
                        )[0]
    test_image = covert_to_tensor(
                        img=load_image(test_image_path),
                        img_size=param['image_size'],
                        num_channels=param['in_channels']
                    ).unsqueeze(0).to(device)
    return test_image
