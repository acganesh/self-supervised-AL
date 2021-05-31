from torch.utils.data import Sampler

import torchvision

class YourSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)

cifar10 = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True)

it = iter(cifar10)

for i, image in enumerate(it):
    img, label = image
    img.save(f"image_{i}_label_{label}.png", format="png")
    import pdb; pdb.set_trace()

    if (i == 10):
        break
    print(i)


