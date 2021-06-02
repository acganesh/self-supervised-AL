import csv
import datetime
import multiprocessing
import os
import pickle
import random
import sys
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torchvision

from PIL import Image
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from collections import Counter
from tqdm import tqdm

from src.data.dataloaders import ImagesDataset
from src.models.model import SelfSupervisedLearner

from config import config_local, config_cluster

if os.environ.get('USER') == 'acganesh':
    DATASET = "STL10"  # or "STL10" or "SVHN" or "CIFAR10"
else:
    DATASET = "CIFAR10"

LR = 3e-4
NUM_CLASSES = 10
NUM_WORKERS = multiprocessing.cpu_count(
) if multiprocessing.cpu_count() < 3 else 3
IMAGE_SIZE = None  # will be populated


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.labels = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:
                self.paths.append(path)
                self.labels.append(int(str(path)[-5]))

        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img), self.labels[index]


def load_config():
    if os.environ.get('USER') == 'acganesh':
        return config_local
    else:
        return config_cluster


C = load_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def get_ckpt_path(model_type):
    assert model_type in C.keys()
    return C['model_type']


def init_model(ds_type='STL10'):
    resnet = torchvision.models.resnet18(pretrained=False)
    if ds_type == 'STL10':
        IMAGE_SIZE = 96
        model = SelfSupervisedLearner(resnet,
                                      image_size=IMAGE_SIZE,
                                      hidden_layer='avgpool',
                                      projection_size=256,
                                      projection_hidden_size=4096,
                                      moving_average_decay=0.99,
                                      ds_type=ds_type,
                                      lr=LR)
        model.load_state_dict(torch.load(C['STL10_WEIGHTS']))
    elif ds_type == 'SVHN':
        IMAGE_SIZE = 32
        model = SelfSupervisedLearner(resnet,
                                      image_size=IMAGE_SIZE,
                                      hidden_layer='avgpool',
                                      projection_size=256,
                                      projection_hidden_size=4096,
                                      moving_average_decay=0.99,
                                      ds_type=ds_type,
                                      lr=LR)
        model.load_state_dict(torch.load(C['SVHN_WEIGHTS']))
    elif ds_type == 'CIFAR10':
        IMAGE_SIZE = 32
        model = SelfSupervisedLearner(resnet,
                                      image_size=IMAGE_SIZE,
                                      hidden_layer='avgpool',
                                      projection_size=256,
                                      projection_hidden_size=4096,
                                      moving_average_decay=0.99,
                                      ds_type=ds_type,
                                      lr=LR)
        model.load_state_dict(torch.load(C['CIFAR10_WEIGHTS']))

    model = model.to(DEVICE)
    print(f"Loaded checkpoint!")
    return model


def init_data(ds_type='STL10'):
    data_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    if ds_type == 'STL10':
        train_dataset = torchvision.datasets.STL10(C['STL10_TRAIN'],
                                                   split='train',
                                                   download=False,
                                                   transform=data_transforms)
        train_loader = DataLoader(train_dataset,
                                  batch_size=5000,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)
        train_imgs, train_labels = next(iter(train_loader))
        train_loader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)

        test_dataset = torchvision.datasets.STL10(C['STL10_TEST'],
                                                  split='test',
                                                  download=False,
                                                  transform=data_transforms)
        test_loader = DataLoader(test_dataset,
                                 batch_size=8000,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))
        test_loader = DataLoader(test_dataset,
                                 batch_size=512,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)

    elif ds_type == 'SVHN':
        train_dataset = torchvision.datasets.SVHN(C['SVHN_EXTRA'],
                                                  split='extra',
                                                  download=False,
                                                  transform=data_transforms)
        train_dataset = torch.utils.data.Subset(train_dataset,
                                                np.arange(50000))
        train_loader = DataLoader(train_dataset,
                                  batch_size=50000,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)
        train_imgs, train_labels = next(iter(train_loader))
        train_loader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)

        test_dataset = torchvision.datasets.SVHN(C['SVHN_TEST'],
                                                 split='test',
                                                 download=False,
                                                 transform=data_transforms)
        test_loader = DataLoader(test_dataset,
                                 batch_size=26032,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))
        test_loader = DataLoader(test_dataset,
                                 batch_size=512,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
    elif ds_type == 'CIFAR10':
        train_dataset = ImagePathDataset(C['BIASED_CIFAR10_TRAIN'])
        train_loader = DataLoader(train_dataset,
                                  batch_size=2750,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)
        train_imgs, train_labels = next(iter(train_loader))
        train_loader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False)

        test_dataset = ImagePathDataset(C['BIASED_CIFAR10_TEST'])
        test_loader = DataLoader(test_dataset,
                                 batch_size=5500,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
        test_imgs, test_labels = next(iter(test_loader))
        test_loader = DataLoader(test_dataset,
                                 batch_size=512,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)

    data_dict = to_data_dict(train_imgs=train_imgs,
                             train_labels=train_labels,
                             test_imgs=test_imgs,
                             test_labels=test_labels)
    loader_dict = {"train_loader": train_loader, "test_loader": test_loader}

    print("Dataset initialized")
    return data_dict, loader_dict


@torch.no_grad()
def featurize_data(model, data_dict, loader_dict):
    D = data_dict
    train_imgs = torch.flatten(D['train_imgs'], start_dim=1)
    test_imgs = torch.flatten(D['test_imgs'], start_dim=1)

    pca = PCA(n_components=512)
    train_projs, test_projs = [], []
    train_embeddings, test_embeddings = [], []
    for train_img, train_label in loader_dict["train_loader"]:
        img = train_img.to(DEVICE)
        cur_projs, cur_embeddings = model.learner.forward(
            img, return_embedding=True)
        train_projs.append(cur_projs)
        train_embeddings.append(cur_embeddings)
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_projs = torch.cat(train_projs, dim=0)

    for test_img, test_label in loader_dict["test_loader"]:
        img = test_img.to(DEVICE)
        cur_projs, cur_embeddings = model.learner.forward(
            img, return_embedding=True)
        test_projs.append(cur_projs)
        test_embeddings.append(cur_embeddings)
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_projs = torch.cat(test_projs, dim=0)

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


def linear_eval(data_dict, features_dict, train_idx, metadata_dict, log=True):
    train_imgs = torch.flatten(data_dict['train_imgs'], start_dim=1)
    train_labels = data_dict['train_labels']
    test_imgs = torch.flatten(data_dict['test_imgs'], start_dim=1)
    test_labels = data_dict['test_labels']

    train_embeddings = features_dict['train_embeddings'].detach().cpu().numpy()
    test_embeddings = features_dict['test_embeddings'].detach().cpu().numpy()

    lr_baseline = LogisticRegression(max_iter=100000)
    lr_baseline.fit(train_imgs[train_idx], train_labels[train_idx])

    lr_baseline_scores = lr_baseline.predict_proba(test_imgs)

    if lr_baseline_scores.shape[1] != NUM_CLASSES:
        lr_baseline_scores = insert_zeros(lr_baseline_scores,
                                          lr_baseline.classes_)

    lr_baseline_preds = lr_baseline.predict(test_imgs)

    lr_byol = LogisticRegression(max_iter=100000)
    lr_byol.fit(train_embeddings[train_idx], train_labels[train_idx])

    lr_byol_scores = lr_byol.predict_proba(test_embeddings)

    if lr_byol_scores.shape[1] != NUM_CLASSES:
        lr_byol_scores = insert_zeros(lr_byol_scores, lr_baseline.classes_)

    lr_byol_preds = lr_byol.predict(test_embeddings)

    prediction_dict = {
        'lr_baseline_scores': lr_baseline_scores,
        'lr_baseline_preds': lr_baseline_preds,
        'lr_byol_scores': lr_byol_scores,
        'lr_byol_preds': lr_byol_preds
    }

    metrics_dict, pr_dict = compute_metrics(data_dict, prediction_dict)
    metrics_dict.update(metadata_dict)
    pr_dict.update(metadata_dict)

    if log:
        print("=" * 80)
        pprint(metrics_dict)

    return metrics_dict, pr_dict


def get_timestamp():
    return datetime.datetime.now().strftime('%m%d%y-%H%M%S')


def multi_class_pr(Y_test, y_score):
    precision = dict()
    recall = dict()
    average_precision = dict()

    n_classes = 10

    Y = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            Y[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])

    pr_dict = {
        'precision': precision,
        'recall': recall,
        'average_precision': average_precision
    }
    return pr_dict


def insert_zeros(scores, mapping):
    mapping = list(mapping)

    new_scores = np.zeros((scores.shape[0], NUM_CLASSES))
    for class_index in range(NUM_CLASSES):
        if class_index not in mapping:
            continue
        else:
            new_scores[:, class_index] = scores[:, mapping.index(class_index)]
    return new_scores


def compute_metrics(data_dict, prediction_dict):
    lr_baseline_scores = prediction_dict['lr_baseline_scores']
    lr_baseline_preds = prediction_dict['lr_baseline_preds']
    lr_byol_preds = prediction_dict['lr_byol_preds']
    lr_byol_scores = prediction_dict['lr_byol_scores']

    test_imgs = data_dict['test_imgs']
    test_labels = data_dict['test_labels']

    lr_baseline_acc = sklearn.metrics.accuracy_score(test_labels,
                                                     lr_baseline_preds)

    lr_baseline_top3_acc = sklearn.metrics.top_k_accuracy_score(
        test_labels, lr_baseline_scores)
    lr_baselise_pr_dict = multi_class_pr(test_labels, lr_baseline_scores)

    lr_baseline_pr = multi_class_pr(test_labels, lr_baseline_scores)

    lr_byol_acc = sklearn.metrics.accuracy_score(test_labels, lr_byol_preds)
    lr_byol_top3_acc = sklearn.metrics.top_k_accuracy_score(
        test_labels, lr_byol_scores)
    lr_byol_pr = multi_class_pr(test_labels, lr_byol_scores)

    metrics_dict = {
        'lr_baseline_acc': lr_baseline_acc,
        'lr_baseline_top3_acc': lr_baseline_top3_acc,
        'lr_byol_acc': lr_byol_acc,
        'lr_byol_top3_acc': lr_byol_top3_acc,
    }

    pr_dict = {'lr_baseline_pr': lr_baseline_pr, 'lr_byol_pr': lr_byol_pr}

    return metrics_dict, pr_dict


def log_metrics(metrics, timestamp):
    timestamp = get_timestamp()
    ds_type = metrics[0]['ds_type']

    if not os.path.exists(C['METRICS_PATH']):
        os.mkdir(C['METRICS_PATH'])
    fpath = os.path.join(C['METRICS_PATH'],
                         f'{timestamp}_{ds_type}_metrics.csv')

    with open(fpath, 'w') as f:
        dict_writer = csv.DictWriter(f, metrics[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(metrics)


def log_prs(pr_all, timestamp):
    timestamp = get_timestamp()
    ds_type = pr_all[0]['ds_type']

    if not os.path.exists(C['METRICS_PATH']):
        os.mkdir(C['METRICS_PATH'])

    fpath = os.path.join(C['METRICS_PATH'], f'{timestamp}_{ds_type}_pr.pkl')

    with open(fpath, 'wb') as f:
        pickle.dump(pr_all, f)


def rand_sample(data_dict, features_dict, num_examples_list):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']
    test_imgs = data_dict['test_imgs']
    test_labels = data_dict['test_labels']
    train_embeddings = features_dict['train_embeddings']
    test_embeddings = features_dict['test_embeddings']

    metrics = []
    pr_list = []
    for num_examples in num_examples_list:
        random_idx = np.random.randint(0,
                                       high=train_imgs.shape[0],
                                       size=num_examples)
        metadata_dict = {
            'sampler_type': 'rand',
            'ds_type': DATASET,
            'num_examples': num_examples
        }

        metrics_dict, pr_dict = linear_eval(data_dict, features_dict,
                                            random_idx, metadata_dict)
        metrics.append(metrics_dict)
        pr_list.append(pr_dict)
    return metrics, pr_list


def kmeans_sample(data_dict, features_dict, num_examples_list):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']
    test_imgs = data_dict['test_imgs']
    test_labels = data_dict['test_labels']
    train_embeddings = features_dict['train_embeddings']
    test_embeddings = features_dict['test_embeddings']

    km = KMeans(n_clusters=10, max_iter=100000)
    km.fit(train_embeddings.detach().cpu().numpy())

    clusters = km.labels_

    counts = Counter(clusters)
    total = train_embeddings.detach().cpu().numpy().shape[0]

    weights = {}
    uniform_prob = 0.1
    for k in counts:
        weights[k] = uniform_prob / (counts[k] / total)

    weights_full = [weights[k] for k in clusters]

    metrics = []
    pr_list = []
    for num_examples in num_examples_list:
        kmeans_idx = random.choices(range(train_imgs.shape[0]),
                                    weights=weights_full,
                                    k=num_examples)

        metadata_dict = {
            'sampler_type': 'kmeans',
            'num_examples': num_examples,
            'ds_type': DATASET
        }
        metrics_dict, pr_dict = linear_eval(data_dict, features_dict,
                                            kmeans_idx, metadata_dict)
        metrics.append(metrics_dict)
        pr_list.append(pr_dict)
    return metrics, pr_list


@torch.no_grad()
def loss_based_ranking(model, data_dict, features_dict, loader_dict,
                       num_examples_list, num_forward_pass):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']

    loss_sum = np.zeros(train_imgs.shape[0])
    loss_sum_squared = np.zeros(train_imgs.shape[0])

    loss_history = {}

    for n in range(num_forward_pass):
        losses_all = []

        for train_img, train_label in loader_dict["train_loader"]:
            img = train_img.to(DEVICE)
            losses = model.learner.forward(img, return_losses=True)
            losses_all.append(losses.detach().cpu().numpy())

        losses_all = np.concatenate(losses_all)

        loss_sum += losses_all
        loss_sum_squared += np.square(losses_all)

        print(f"Progress: {n+1}/{num_forward_pass} forward passes complete")

    loss_means = loss_sum / num_forward_pass
    loss_stds = np.sqrt(loss_sum_squared / num_forward_pass -
                        np.square(loss_means))

    metrics = []
    pr_list = []
    for num_examples in num_examples_list:
        ### Mean eval ###
        metadata_dict = {
            'sampler_type': 'loss_based_mean',
            'num_examples': num_examples,
            'ds_type': DATASET
        }
        idx = np.argsort(-loss_means)
        mean_subset = idx[:num_examples]
        print("Mean Loss Eval:")
        metrics_dict, pr_dict = linear_eval(data_dict, features_dict,
                                            mean_subset, metadata_dict)
        metrics.append(metrics_dict)
        pr_list.append(pr_dict)

        ### Stdev eval ###
        metadata_dict = {
            'sampler_type': 'loss_based_std',
            'num_examples': num_examples,
            'ds_type': DATASET
        }
        idx = np.argsort(-loss_stds)
        std_subset = idx[:num_examples]
        print("STD Loss Eval:")
        metrics_dict, pr_dict = linear_eval(data_dict, features_dict,
                                            std_subset, metadata_dict)
        metrics.append(metrics_dict)
        pr_list.append(pr_dict)

    return metrics, pr_list


def grad_based_ranking(model, data_dict, features_dict, loader_dict,
                       num_examples_list):
    train_imgs = data_dict['train_imgs']
    train_labels = data_dict['train_labels']
    train_embeddings = features_dict['train_embeddings']

    train_norms = np.zeros(train_imgs.shape[0])

    # Global img index
    j = 0

    model.eval()
    pbar = tqdm(total=train_embeddings.shape[0])
    for train_img, train_label in loader_dict["train_loader"]:
        img = train_img.to(DEVICE)
        for cur_img in range(img.shape[0]):
            model.zero_grad()
            proj1, proj2, loss = model.learner.forward(
                img[cur_img].unsqueeze(0),
                return_embedding=False,
                return_projection=False,
                return_losses=False,
                return_losses_and_embeddings=True)
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    train_norms[j] += torch.sum(torch.square(param.grad))
            train_norms[j] = np.sqrt(train_norms[j])
            j += 1
            pbar.update(1)

    pbar.close()

    # Ensure it is zeroed
    model.zero_grad()

    # Select
    metrics = []
    pr_list = []
    for num_examples in num_examples_list:
        idx = np.argsort(-train_norms)

        grad_subset = idx[:num_examples]
        print("Grad Based Eval:")

        metadata_dict = {
            'sampler_type': 'grad_based',
            'num_examples': num_examples,
            'ds_type': DATASET
        }
        metrics_dict, pr_dict = linear_eval(data_dict, features_dict,
                                            grad_subset, metadata_dict)
        metrics.append(metrics_dict)
        pr_list.append(pr_dict)
    return metrics, pr_list

    # Angle selection
    # angles = np.zeros(train_imgs.shape[0])
    # mean_grad = np.mean(train_grads, axis=0)
    # v_1 = mean_grad / np.linalg.norm(mean_grad)

    # for index in range(train_imgs.shape[0]):
    #     v_2 = train_grads[index] / np.linalg.norm(train_grads[index])
    #     angles[index] = np.arccos(np.dot(v_1, v_2))

    # idx = np.argsort(-angles)
    # subset = idx[:n_examples]
    # train_imgs_subset_angles = train_imgs[subset]
    # train_labels_subset_angles = train_labels[subset]


def main():
    model = init_model(DATASET)

    if os.environ.get('USER') == 'acganesh':
        with open("cache/data_dict.pkl", 'rb') as f:
            data_dict = pickle.load(f)

        with open("cache/features_dict.pkl", 'rb') as f:
            features_dict = pickle.load(f)
    else:
        data_dict, loader_dict = init_data(DATASET)
        features_dict = featurize_data(model, data_dict, loader_dict)

    print("Data and features loaded!")

    num_examples_list = [5, 10]

    metrics_all = []
    pr_all = []

    metrics, pr = rand_sample(data_dict,
                              features_dict,
                              num_examples_list=num_examples_list)
    metrics_all += metrics
    pr_all += pr

    metrics, pr = kmeans_sample(data_dict,
                                features_dict,
                                num_examples_list=num_examples_list)
    metrics_all += metrics
    pr_all += pr

    if os.environ.get('USER') != 'acganesh':
        metrics, pr = loss_based_ranking(model,
                                         data_dict,
                                         features_dict,
                                         loader_dict,
                                         num_examples_list=num_examples_list,
                                         num_forward_pass=5)
        metrics_all += metrics
        pr_all += pr

        metrics, pr = grad_based_ranking(model,
                                         data_dict,
                                         features_dict,
                                         loader_dict,
                                         num_examples_list=num_examples_list)
        metrics_all += metrics
        pr_all += pr

    timestamp = get_timestamp()
    log_metrics(metrics_all, timestamp)
    log_prs(pr_all, timestamp)


if __name__ == '__main__':
    main()
