import numpy as np
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
        transforms.GaussianBlur(3), # May need to change depending on input size (?)
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
    