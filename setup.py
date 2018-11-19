"""Package setup script
"""
import sys
from setuptools import setup

requirements = [
    'ipython',
    'tqdm',
    'numpy',
    'pillow',
    'matplotlib',
    'seaborn',
    'pandas',
    'pyarrow',
    'torch',
    'torchvision',
    'nltk'
]

if sys.version_info < (3, 7):
    requirements.append('dataclasses')


test_requirements = [
    'tox',
    'pytest',
    'pytest-pep8',
    'pytest-xdist',
    'pytest-cov',
    'pytest-timeout'
]


setup(
    name='smartai',
    version='0.0.999',
    description='Smart Python Library for Artificial Intelligence',
    long_description="Can we let the data build models by itself?",
    url='https://github.com/yajiez/smartai',
    author='Yajie',
    packages=['src/smartai'],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    extras_require={
        'tests': test_requirements
    }
)
