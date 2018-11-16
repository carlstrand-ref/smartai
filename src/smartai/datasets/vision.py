import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from . import DATASET_ROOT_DIR


def load_mnist(
        root=None, train=True, transform=None, target_transform=None, download=True,
        return_loader=True, batch_size=16, num_workers=0,  shuffle=False):
    root = os.path.join(DATASET_ROOT_DIR, 'mnist') if not root else root
    dataset = MNIST(root, train=train, transform=transform, target_transform=target_transform, download=download)
    if return_loader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    else:
        return dataset
