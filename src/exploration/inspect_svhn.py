from config import config_local, config_cluster
from subset_selection import load_config

def inspect_svhn():
    C = load_config()

    train_dataset = torchvision.datasets.SVHN(C['SVHN_EXTRA'],
					      split='extra',
					      download=True,
					      transform=data_transforms)
    train_dataset = torch.utils.data.Subset(train_dataset,
					    np.arange(50000))
    train_loader = DataLoader(train_dataset,
			      batch_size=50000,
			      num_workers=NUM_WORKERS,
			      shuffle=False)
    train_imgs, train_labels = next(iter(train_loader))

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    inspect_svhn()
