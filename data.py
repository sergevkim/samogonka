import typing as tp
import pytorch_lightning as ptl
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


class DenoiserDataModule(ptl.LightningDataModule):
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.path = dataset_path
        self.batch_size = batch_size
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              torchvision.transforms.RandomHorizontalFlip(),
                                              torchvision.transforms.RandomInvert(p=0.1),
                                              torchvision.transforms.RandomGrayscale(),
                                              torchvision.transforms.GaussianBlur(3),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.path, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.path, train=False, download=True)

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_set = torchvision.datasets.CIFAR10(root=self.path, train=True, transform=self.transforms)
        self.val_set = torchvision.datasets.CIFAR10(root=self.path, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=1)