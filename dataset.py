# https://github.com/AntixK/PyTorch-VAE/blob/master/dataset.py
import warnings
import torch
import os
import utility as util
from typing import List, Union, Optional, Sequence,Callable
from pathlib import Path
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, 
                 data_path:str,
                 split: str,
                #  transform: Callable,
                 target_size:int, 
                 num_channels:int=1):
        
        self.root_dir = Path(data_path)
        # self.transforms = transform
        self.target_size = target_size
        self.num_channels = num_channels
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        imgs = sorted([f for f in self.root_dir.iterdir() if f.suffix in supported_formats])
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx:int):
        img = default_loader(self.imgs[idx])
        
        img = util.covert_rgba(img=img,
                               img_size=self.target_size, 
                               num_channels=self.num_channels)
        img = transforms.ToTensor()(img)
        # if self.transforms is not None:
        #     img = self.transforms(img)
        
        return img # dummy datat to prevent breaking 
        
class VAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        img_size: Union[int, Sequence[int]] = (256, 256),
        num_channels: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.img_size = img_size
        self.channels = num_channels
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
        
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            #   transforms.CenterCrop(256),
                                            #   transforms.Resize(self.img_size),
                                              transforms.ToTensor()])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(256),
                                            # transforms.Resize(self.img_size),
                                            transforms.ToTensor()])
        
        self.train_dataset = CustomDataset(
            data_path=self.data_dir,
            split='train',
            # transform=train_transforms,
            target_size=self.img_size,
            num_channels=self.channels
        )
        
        self.val_dataset = CustomDataset(
            data_path=self.data_dir,
            split='test',
            # transform=val_transforms,
            target_size=self.img_size,
            num_channels=self.channels
        )
        
        # self.test_dataset = CustomDataset(
        #     data_path=self.data_dir,
        #     target_size=self.img_size,
        #     num_channels=self.channels
        # )
        
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