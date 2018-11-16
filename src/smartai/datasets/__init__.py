import os

DATASET_ROOT_DIR = os.path.expanduser('~/.smartai/datasets')
os.makedirs(DATASET_ROOT_DIR, exist_ok=True)

from .vision import load_mnist

__all__ = [
    'DATASET_ROOT_DIR',
    'load_mnist'
]
