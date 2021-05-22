import os
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset, input_size=(32, 32), batch_size=32):
    jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2) 
    train_transformations = transforms.Compose([
        transforms.RandomResizedCrop(size=input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur((0.1 * input_size[0], 0.1 * input_size[1])), 
        transforms.ToTensor(),
        ])
    
    eval_transformations = transforms.Compose([
        transforms.CenterCrop(size=input_size)
        ])
    
    
    # TODO: Mean and STD normalization after augmentation :/
    
    if dataset == "cifar100":               
        train_dataset = datasets.CIFAR100("data/cifar100/", train=True, download=True, transform=train_transformations)
        test_dataset = datasets.CIFAR100("data/cifar100/", train=False, download=True, transform=eval_transformations)
    else:
        raise NotImplementedError
        
    train_dataloder = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloder = DataLoader(test_dataset, batch_size, shuffle=False)
        
    return train_dataloder, test_dataloder    

IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size, train):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(expand_greyscale)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)