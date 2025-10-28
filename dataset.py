from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import utility as util

class CustomDataset(Dataset):
    def __init__(self, image_paths:str, dataset_size:int, target_size:tuple=(256,256)):
        self.image_paths = image_paths
        self.dataset_size = dataset_size
        self.target_size = target_size
        self.image_size_flat = target_size[0] * target_size[1]
        try:
            self.base_image = Image.open(image_paths).convert('L')
        except FileNotFoundError:
            raise FileNotFoundError(f"Not Found: {image_paths}")
        self.transform = transforms.Compose([
            # --- Geometric Augmentations  ---
            
            # --- Random Resized Crop ---
            transforms.RandomResizedCrop(self.target_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            
            # --- Flipping ---
            transforms.RandomHorizontalFlip(p=0.5),
            
            # --- Rotation ---
            transforms.RandomRotation(degrees=10),
            
            # --- Color/Photometric Augmentations ---
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            
            # --- Conversion ---
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        augmented_tensor = self.transform(self.base_image)
        return augmented_tensor.view(1, self.image_size_flat)

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(
            self,      
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers 
        )
