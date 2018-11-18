# -*- coding: utf-8 -*-
import os

PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, os.pardir, os.pardir))
DATASET_ROOT_DIR = os.path.expanduser('~/.smartai/datasets')
os.makedirs(DATASET_ROOT_DIR, exist_ok=True)

from . import datasets
from . import models
from . import utils


__version__ = '0.0.1.dev'

__all__ = [
    'datasets',
    'models',
    'utils'
]
