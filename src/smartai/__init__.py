# -*- coding: utf-8 -*-
import os

__version__ = '0.0.1.dev'

__all__ = [
    'datasets',
    'models',
    'utils'
]


PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, os.pardir, os.pardir))
