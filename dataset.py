# https://github.com/AntixK/PyTorch-VAE/blob/master/dataset.py
import warnings
import torch
import os
import utility as util
from typing import List, Union, Optional, Sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, data_path:str, target_size:int, num_channels:int=1):
        self.root_dir = data_path
        self.image_paths = []
        self.target_size = target_size
        self.image_size_flat = target_size * target_size
        self.num_channels = num_channels

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # Gather all image paths
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"No directory found: {data_path}")
            
        for fname in os.listdir(data_path):
            if fname.lower().endswith(supported_formats):
                self.image_paths.append(os.path.join(data_path, fname))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx:int):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path)
        except Exception as e:
            
            warnings.warn(f"Error loading image {img_path}: {e}. Returning black image.")
            return torch.zeros(self.num_channels, 
                               self.target_size[0], 
                               self.target_size[1])
        
        return util.preprocess_image(image, 
                                     num_channels=self.num_channels,
                                     image_size=self.target_size)
        
class VAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_channels: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.channels = num_channels
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
                
        self.train_dataset = CustomDataset(
            data_path=self.data_dir,
            target_size=self.patch_size,
            num_channels=self.channels
        )
        
        self.val_dataset = CustomDataset(
            data_path=self.data_dir,
            target_size=self.patch_size,
            num_channels=self.channels
        )
        
        self.test_dataset = CustomDataset(
            data_path=self.data_dir,
            target_size=self.patch_size,
            num_channels=self.channels
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )