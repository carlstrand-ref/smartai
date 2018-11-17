import os

DATASET_ROOT_DIR = os.path.expanduser('~/.smartai/datasets')
os.makedirs(DATASET_ROOT_DIR, exist_ok=True)

from .vision import load_mnist
from .vision import load_emnist
from .vision import load_fashion_mnist
from .vision import load_cifar10
from .vision import load_cifar100

__all__ = [
    'DATASET_ROOT_DIR',
    'load_mnist',
    'load_emnist',
    'load_fashion_mnist',
    'load_cifar10',
    'load_cifar100'
]
