"""Package setup script
"""

from setuptools import setup

setup(
    name='smartai',
    version='0.0.999',
    description='Smart Python Library for Artificial Intelligence',
    long_description="Stop building models for your data, let your data build models by itself.",
    url='https://github.com/yajiez/smartai',
    author='Yajie',
    packages=['src/smartai'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'ipython',
        'tqdm',
        'numpy',
        'matplotlib',
        'pandas',
        'pyarrow',
        'torch',
        'torchvision',
        'nltk'
    ],
    extras_require={
        'tests': [
            'tox',
            'pytest',
            'pytest-pep8',
            'pytest-xdist',
            'pytest-cov',
            'pytest-timeout'
        ]
    }
)
