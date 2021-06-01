
import torchvision
import numpy as np
from collections import defaultdict
from tqdm import tqdm
num_classes = 10

save_dir = '/scratch/users/avento/datasets/biased_cifar10/test_images/'
cifar10 = torchvision.datasets.CIFAR10(root='/scratch/users/avento/datasets/raw_files/cifar10_raw', train=False, download=True)
it = iter(cifar10)
num_images = 0
indices_to_use = {}
cur_class_index = defaultdict(int)
channel_sum = np.zeros(3)
channel_sum_squared = np.zeros(3)
num_pixels = 0
for class_index in range(num_classes):
    num_to_sample = (class_index + 1) * 100
    index_to_sample = np.linspace(0, 999, num=num_to_sample, dtype=int)
    indices_to_use[class_index] = set(index_to_sample)
for i, image in tqdm(enumerate(it)):
    img, label = image
    cur_index = cur_class_index[label]
    if cur_index in indices_to_use[label]:
        save_location = save_dir + f"image_{i}_label_{label}.png"
        img.save(save_location, format="png")
    cur_class_index[label] = cur_class_index[label] + 1