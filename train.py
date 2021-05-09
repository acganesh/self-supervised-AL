import multiprocessing
import sys

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader

from models.model import SelfSupervisedLearner
from data.dataloaders import ImagesDataset

BATCH_SIZE = 32
EPOCHS     = 1000
LR         = 3e-4
IMAGE_SIZE = 96 # Change this depending on dataset
NUM_GPUS= 0 # Change this depending on host
NUM_WORKERS = multiprocessing.cpu_count()


def main(argv):
    print("Args: ", argv)
    resnet = torchvision.models.resnet18(pretrained=False)
    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    if (argv[1] == "--train"):
        #ds = torchvision.datasets.STL10("./STL10", download=True)
        ds = ImagesDataset("./data/unlabeled_images", IMAGE_SIZE, train=True)
        train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

        trainer = pl.Trainer(
            gpus = NUM_GPUS,
            max_epochs = 1,
            accumulate_grad_batches = 1,
            sync_batchnorm = True
        )

        trainer.fit(model, train_loader)
    elif argv[1] == "--load":
        torch.load(argv[2])
        print("Loaded checkpoint from ", argv[2])

        #TODO: for some reason labels don't exist in my wget data 
        #ds = ImagesDataset("./data/test_images", IMAGE_SIZE, train=False)
        data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_dataset = torchvision.datasets.STL10('./data/test_split', split='test', download=True,
                               transform=data_transforms)
        test_loader = DataLoader(ds, batch_size=8000, num_workers=NUM_WORKERS, shuffle=False)

        imgs = next(iter(test_loader))

        import pdb; pdb.set_trace()

        

if __name__ == '__main__':
    main(sys.argv)