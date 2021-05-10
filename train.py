import multiprocessing
import sys

import pytorch_lightning as pl
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from models.model import SelfSupervisedLearner
from data.dataloaders import ImagesDataset

BATCH_SIZE = 256
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
        moving_average_decay = 0.99,
        lr = LR
    )

    if (argv[1] == "--train"):
        #ds = torchvision.datasets.STL10("./STL10", download=True)
        ds = ImagesDataset("./data/unlabeled_images", IMAGE_SIZE, train=True)
        train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

        trainer = pl.Trainer(
            gpus = NUM_GPUS,
            max_epochs = 10,
            accumulate_grad_batches = 1,
            sync_batchnorm = True
        )

        trainer.fit(model, train_loader)

        import pdb; pdb.set_trace()
    elif argv[1] == "--load":
        torch.load(argv[2])
        print("Loaded checkpoint from ", argv[2])

        #TODO: for some reason labels don't exist in my wget data 
        #ds = ImagesDataset("./data/test_images", IMAGE_SIZE, train=False)
        data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_dataset = torchvision.datasets.STL10('./data/train_split', split='train', download=False,
                               transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=5000, num_workers=NUM_WORKERS, shuffle=False)


        train_imgs, train_labels = next(iter(train_loader))
        print("Train loading done")

        test_dataset = torchvision.datasets.STL10('./data/test_split', split='test', download=False, transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=8000, num_workers=NUM_WORKERS, shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))
        print("Test loading done")

        train_projs, train_embeddings = model.learner.forward(train_imgs, return_embedding=True)
        test_projs, test_embeddings = model.learner.forward(test_imgs, return_embedding=True)

        print("got embeddings")

        train_imgs = torch.flatten(train_imgs, start_dim=1)
        test_imgs = torch.flatten(test_imgs, start_dim=1)
        pca = PCA(n_components=256)
        train_imgs_pca = pca.fit_transform(train_imgs)
        test_imgs_pca = pca.transform(test_imgs)

        lr_baseline = LogisticRegression(max_iter=100000)
        baseline_preds = lr_baseline.fit(train_imgs_pca, train_labels)

        baseline_preds = lr_baseline.predict_proba(test_imgs_pca)
        baseline_classes = lr_baseline.predict(test_imgs_pca)
        baseline_acc = sklearn.metrics.accuracy_score(test_labels, baseline_classes)

        lr_byol = LogisticRegression(max_iter=100000)
        lr_byol.fit(train_projs.detach().numpy(), train_labels)

        byol_preds = lr_byol.predict_proba(test_projs.detach().numpy())
        byol_classes = lr_byol.predict(test_projs.detach().numpy())
        byol_acc = sklearn.metrics.accuracy_score(test_labels, byol_classes)

        import pdb; pdb.set_trace()

        

if __name__ == '__main__':
    main(sys.argv)