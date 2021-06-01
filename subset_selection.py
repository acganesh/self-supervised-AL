import multiprocessing
import random
import sys

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torchvision

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from collections import Counter

from src.data.dataloaders import ImagesDataset
from src.models.model import SelfSupervisedLearner

BATCH_SIZE = 256
EPOCHS = 1000
LR = 3e-4
IMAGE_SIZE = 96  # Change this depending on dataset
NUM_GPUS = 0  # Change this depending on host
NUM_WORKERS = multiprocessing.cpu_count()


def to_data_dict(train_imgs, train_labels, test_imgs, test_labels):
    data_dict = {
        'train_imgs': train_imgs,
        'train_labels': train_labels,
        'test_imgs': test_imgs,
        'test_labels': test_labels
    }
    return data_dict


def to_features_dict(train_imgs_pca, test_imgs_pca, train_projs, test_projs,
                     train_embeddings, test_embeddings):
    data_dict = {
        'train_imgs_pca': train_imgs_pca,
        'test_imgs_pca': test_imgs_pca,
        'train_projs': train_projs,
        'test_projs': test_projs,
        'train_embeddings': train_embeddings,
        'test_embeddings': test_embeddings
    }
    return data_dict


# TODO: index the checkpoints by key
def init_model(ckpt_path):
    resnet = torchvision.models.resnet18(pretrained=False)
    model = SelfSupervisedLearner(resnet,
                                  image_size=IMAGE_SIZE,
                                  hidden_layer='avgpool',
                                  projection_size=256,
                                  projection_hidden_size=4096,
                                  moving_average_decay=0.99,
                                  lr=LR)
    model.load_state_dict(torch.load(ckpt_path))
    print(f"Loaded checkpoint from {ckpt_path}")
    return model


def init_data(ds_type='STL10'):
    data_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    if ds_type == 'STL10':
        train_dataset = torchvision.datasets.STL10(
            './dataset/stl10/train_split',
            split='train',
            download=False,
            transform=data_transforms)
        train_loader = DataLoader(train_dataset,
                                  batch_size=5000,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)
        train_imgs, train_labels = next(iter(train_loader))

        test_dataset = torchvision.datasets.STL10('./dataset/stl10/test_split',
                                                  split='test',
                                                  download=False,
                                                  transform=data_transforms)
        test_loader = DataLoader(test_dataset,
                                 batch_size=8000,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))

    elif ds_type == 'SVHN':
        train_dataset = torchvision.datasets.SVHN('./dataset/svhn/extra',
                                                  split='extra',
                                                  download=False,
                                                  transform=data_transforms)
        train_loader = DataLoader(train_dataset,
                                  batchsize=100000,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)
        train_imgs, train_labels = next(iter(train_loader))

        test_dataset = torchvision.datasets.SVHN('./dataset/svhn/test',
                                                 split='test',
                                                 download=False,
                                                 transform=data_transforms)
        test_loader = DataLoader(test_dataset,
                                 batch_size=26032,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))

    data_dict = to_data_dict(train_imgs=train_imgs,
                             train_labels=train_labels,
                             test_imgs=test_imgs,
                             test_labels=test_labels)

    print("Dataset initialized")
    return data_dict


def featurize_data(model, data_dict):
    D = data_dict
    train_imgs = torch.flatten(D['train_imgs'], start_dim=1)
    test_imgs = torch.flatten(D['test_imgs'], start_dim=1)

    pca = PCA(n_components=512)
    train_projs, train_embeddings = model.learner.forward(
        D['train_imgs'], return_embedding=True)
    test_projs, test_embeddings = model.learner.forward(D['test_imgs'],
                                                        return_embedding=True)

    train_imgs_pca = pca.fit_transform(
        torch.flatten(D['train_imgs'], start_dim=1))
    test_imgs_pca = pca.transform(torch.flatten(D['test_imgs'], start_dim=1))

    features_dict = to_features_dict(train_imgs_pca=train_imgs_pca,
                                     test_imgs_pca=test_imgs_pca,
                                     train_projs=train_projs,
                                     test_projs=test_projs,
                                     train_embeddings=train_embeddings,
                                     test_embeddings=test_embeddings)
    print("Features dict initialized")
    return features_dict


def get_predictions(data_dict, features_dict):
    D = data_dict
    F = features_dict

    lr_baseline = LogisticRegression(max_iter=100000)
    baseline_preds = lr_baseline.fit(F['train_imgs_pca'], D['train_labels'])

    baseline_preds = lr_baseline.predict_proba(F['test_imgs_pca'])
    baseline_classes = lr_baseline.predict(F['test_imgs_pca'])
    baseline_acc = sklearn.metrics.accuracy_score(D['test_labels'],
                                                  baseline_classes)

    lr_byol = LogisticRegression(max_iter=100000)
    lr_byol.fit(F['train_embeddings'].detach().numpy(), D['train_labels'])

    byol_preds = lr_byol.predict_proba(F['test_embeddings'].detach().numpy())
    byol_classes = lr_byol.predict(F['test_embeddings'].detach().numpy())
    byol_acc = sklearn.metrics.accuracy_score(D['test_labels'], byol_classes)

    return baseline_preds, baseline_acc, byol_preds, byol_acc


def rand_sample(data_dict, features_dict):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']
    test_imgs = data_dict['test_imgs']
    test_labels = data_dict['test_labels']
    train_embeddings = features_dict['train_embeddings']
    test_embeddings = features_dict['test_embeddings']

    random_idx = np.random.randint(0, high=train_imgs.shape[0], size=30)

    embeddings_subset = train_embeddings.detach().numpy()[random_idx]
    train_labels_subset = train_labels[random_idx]

    lr_rand = LogisticRegression(max_iter=100000)
    lr_rand.fit(embeddings_subset, train_labels_subset)

    rand_preds = lr_rand.predict(test_embeddings.detach().numpy())
    rand_acc = sklearn.metrics.accuracy_score(test_labels, rand_preds)

    lr_baseline = LogisticRegression(max_iter=100000)
    lr_baseline.fit(torch.flatten(train_imgs[random_idx], start_dim=1), train_labels_subset)

    lr_baseline_preds = lr_baseline.predict(torch.flatten(test_imgs, start_dim=1))
    lr_baseline_acc = sklearn.metrics.accuracy_score(test_labels,
                                                     lr_baseline_preds)

    # rand_acc: accuracy of train lr on random byol embeddings
    # lr_baseline_acc: accuracy of training lr on random images
    print("lr baseline: ", lr_baseline_acc)
    print("random embeddings: ", rand_acc)

    return rand_acc, lr_baseline


def kmeans_sample(data_dict, features_dict):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']
    test_imgs = data_dict['test_imgs']
    test_labels = data_dict['test_labels']
    train_embeddings = features_dict['train_embeddings']
    test_embeddings = features_dict['test_embeddings']

    km = KMeans(n_clusters=10, max_iter=100000)
    km.fit(train_embeddings.detach().numpy())

    clusters = km.labels_

    counts = Counter(clusters)
    total = train_embeddings.detach().numpy().shape[0]

    weights = {}
    uniform_prob = 0.1
    for k in counts:
        weights[k] = uniform_prob / (counts[k] / total)

    weights_full = [weights[k] for k in clusters]

    kmeans_idx = random.choices(range(train_imgs.shape[0]),
                                weights=weights_full,
                                k=30)

    embeddings_subset = train_embeddings.detach().numpy()[kmeans_idx]
    train_labels_subset = train_labels[kmeans_idx]

    lr_km = LogisticRegression(max_iter=100000)
    lr_km.fit(embeddings_subset, train_labels_subset)

    km_preds = lr_km.predict(test_embeddings.detach().numpy())
    km_acc = sklearn.metrics.accuracy_score(test_labels, km_preds)

    lr_baseline = LogisticRegression(max_iter=100000)
    lr_baseline.fit(torch.flatten(train_imgs[kmeans_idx], start_dim=1), train_labels_subset)

    lr_baseline_preds = lr_baseline.predict(torch.flatten(test_imgs, start_dim=1))
    lr_baseline_acc = sklearn.metrics.accuracy_score(test_labels,
                                                     lr_baseline_preds)

    print("km: ", km_acc)
    print("lr baseline acc:", lr_baseline_acc)

    return km_acc, lr_baseline_acc


def loss_based_ranking(model, data_dict, features_dict, n_examples, mode='mean'):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']

    # patched BYOL lib to return losses directly
    losses = model.learner.forward(train_imgs[0], return_losses=True)

    means = np.mean(losses)
    stds = np.std(losses)

    if mode == 'mean':
        idx = np.argsort(means)
        subset = idx[:n_examples]
        train_imgs_subset = train_imgs[subset]
        train_labels_subset = train_labels[subset]
    elif mode == 'std':
        idx = np.argsort(stds)
        subset = idx[:n_examples]
        train_imgs_subset = train_imgs[subset]
        train_labels_subset = train_labels[subsets]

    return train_imgs_subset, train_labels_subset

def main():
    # TODO: convert to flag
    ckpt_path = './ckpt/learner_0510_v100.pt'
    model = init_model(ckpt_path=ckpt_path)
    data_dict = init_data()
    features_dict = featurize_data(model, data_dict)
    rand_sample(data_dict, features_dict)
    kmeans_sample(data_dict, features_dict)
    loss_based_ranking(model, data_dict, features_dict, n_examples=10, mode='mean')


if __name__ == '__main__':
    main()
