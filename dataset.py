
import utility as util
from typing import List, Union, Optional, Sequence
from pathlib import Path
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, 
                 data_path:str,
                 transform:transforms,
                 split: str,
                 split_set: float=0.8,
                 target_size:int=128, 
                 num_channels:int=1):
        
        self.root_dir = Path(data_path)
        self.transform = transform
        self.target_size = target_size
        self.num_channels = num_channels
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        imgs = sorted([f for f in self.root_dir.iterdir() if f.suffix in supported_formats])
        self.imgs = imgs[:int(len(imgs) * split_set)] if split == "train" else imgs[int(len(imgs) * split_set):]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx:int):
        img = default_loader(self.imgs[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        return img
        
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
        transform_list = []
        if self.channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img))
        transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        transform_list.append(transforms.ToTensor())
        
        train_transforms = transforms.Compose(transform_list)
        
        val_transforms = transforms.Compose(transform_list)
        
        self.train_dataset = CustomDataset(
            data_path=self.data_dir,
            transform=train_transforms,
            split='train',
            target_size=self.img_size,
            num_channels=self.channels
        )
        
        self.val_dataset = CustomDataset(
            data_path=self.data_dir,
            transform=val_transforms,
            split='test',
            target_size=self.img_size,
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
