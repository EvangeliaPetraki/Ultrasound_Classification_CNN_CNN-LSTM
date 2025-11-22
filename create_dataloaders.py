from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlbumentationsDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.dataset = datasets.ImageFolder(root=rootDir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label
    
    @property
    def classes(self):
        return self.dataset.classes 

def get_dataloader(rootDir, transforms, batchSize, shuffle=True):
    ds = AlbumentationsDataset(rootDir, transform=transforms)
    loader = DataLoader(ds, batch_size=batchSize, shuffle=shuffle, num_workers=0, pin_memory=True if DEVICE == "cuda" else False)
    return ds, loader

