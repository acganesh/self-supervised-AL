import multiprocessing
import socket
import sys

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torchvision

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.data.dataloaders import ImagesDataset
from src.models.model import SelfSupervisedLearner

BATCH_SIZE = 512
EPOCHS = 1000
LR = 3e-4
IMAGE_SIZE = 96  # Change this depending on dataset
NUM_GPUS = 0  # Change this depending on host
NUM_WORKERS = multiprocessing.cpu_count()


def main(argv):
    print("Args: ", argv)
    resnet = torchvision.models.resnet50(pretrained=False)
    model = SelfSupervisedLearner(resnet,
                                  image_size=IMAGE_SIZE,
                                  hidden_layer='avgpool',
                                  projection_size=256,
                                  projection_hidden_size=4096,
                                  moving_average_decay=0.99,
                                  lr=LR)

    if (argv[1] == "--train"):
        ds = ImagesDataset("./dataset/unlabeled_images",
                           IMAGE_SIZE,
                           train=True)
        train_loader = DataLoader(ds,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

        trainer = pl.Trainer(gpus=NUM_GPUS,
                             max_epochs=10,
                             accumulate_grad_batches=1,
                             sync_batchnorm=True,
                             default_root_dir="./checkpoints")

        trainer.fit(model, train_loader)

        import pdb
        pdb.set_trace()

if __name__ == '__main__':
    main(sys.argv)
