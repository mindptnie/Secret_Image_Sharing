import warnings
import torch
import os
import utility as util
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir:str, target_size:tuple=(256,256)):
        self.root_dir = root_dir
        self.image_paths = []
        self.target_size = target_size
        # CHANGED: No longer using image_size_flat since we keep spatial dimensions
        print(root_dir)

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # Gather all image paths
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"No directory found: {root_dir}")
            
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(supported_formats):
                self.image_paths.append(os.path.join(root_dir, fname))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            # CHANGED: Convert to 'RGB' instead of 'L' (grayscale)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {e}. Returning black image.")
            # CHANGED: Return 3-channel black image
            return torch.zeros(3, self.target_size[0], self.target_size[1])
        
        return util.preprocess_image(image, image_size=self.target_size)
    
    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(
            self,      
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers 
        )