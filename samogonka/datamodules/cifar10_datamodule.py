from pathlib import Path
from typing import Callable, Optional, Union

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        raw_cifar: Union[CIFAR10, Subset], # CIFAR10 or subset of it
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.raw_cifar = raw_cifar
        self.transform = transform

    def __len__(self) -> int:
        return len(self.raw_cifar)

    def __getitem__(self, idx):
        image, label = self.raw_cifar[idx]
        # be careful! image is not a numpy array

        if self.transform is not None:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = './data/',
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = False,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        val_size: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.val_size = val_size

    def prepare_data(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            trainval_raw_dataset = CIFAR10(root=self.data_dir, train=True)
            train_indices, val_indices = train_test_split(
                np.arange(len(trainval_raw_dataset)),
                test_size=self.val_size,
            ) # TODO stratify
            train_raw_subset = Subset(trainval_raw_dataset, train_indices)
            val_raw_subset = Subset(trainval_raw_dataset, val_indices)

            train_transforms = self.default_transforms() \
                if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() \
                if self.val_transforms is None else self.val_transforms
            self.train_dataset = \
                CIFAR10Dataset(train_raw_subset, transform=train_transforms)
            self.val_dataset = \
                CIFAR10Dataset(val_raw_subset, transform=val_transforms)

        if stage == 'test' or stage is None:
            test_transforms = self.default_transforms() \
                if self.test_transforms is None else self.test_transforms
            test_raw_dataset = CIFAR10(root=self.data_dir, train=False)
            self.test_dataset = \
                CIFAR10Dataset(test_raw_dataset, transform=test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def default_transforms(self) -> Callable:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
