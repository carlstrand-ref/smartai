import os
import pickle
from torch.utils.data import DataLoader
import torchvision
from .. import DATASET_ROOT_DIR


def unpickle_cifar_meta(meta_file):
    with open(meta_file, 'rb') as fo:
        meta = pickle.load(fo)
    return meta


cifar100_classes_coarse_to_fine = {
    "aquatic mammals":                "beaver, dolphin, otter, seal, whale",
    "fish":                           "aquarium fish, flatfish, ray, shark, trout",
    "flowers":                        "orchid, poppy, rose, sunflower, tulip",
    "food containers":                "bottle, bowl, can, cup, plate",
    "fruit and vegetables":           "apple, mushroom, orange, pear, sweet pepper",
    "household electrical devices":   "clock, keyboard, lamp, telephone, television",
    "household furniture":            "bed, chair, couch, table, wardrobe",
    "insects":                        "bee, beetle, butterfly, caterpillar, cockroach",
    "large carnivores":               "bear, leopard, lion, tiger, wolf",
    "large man-made outdoor things":  "bridge, castle, house, road, skyscraper",
    "large natural outdoor scenes":   "cloud, forest, mountain, plain, sea",
    "large omnivores and herbivores": "camel, cattle, chimpanzee, elephant, kangaroo",
    "medium-sized mammals":           "fox, porcupine, possum, raccoon, skunk",
    "non-insect invertebrates":       "crab, lobster, snail, spider, worm",
    "people":                         "baby, boy, girl, man, woman",
    "reptiles":                       "crocodile, dinosaur, lizard, snake, turtle",
    "small mammals":                  "hamster, mouse, rabbit, shrew, squirrel",
    "trees":                          "maple tree, oak tree, palm tree, pine tree, willow tree",
    "vehicles 1":                     "bicycle, bus, motorcycle, pickup truck, train",
    "vehicles 2":                     "lawn mower, rocket, streetcar, tank, tractor"
}


def cifar100_fine_to_coarse():
    fine_to_coarse = {}
    for coarse_label, fine_labels in cifar100_classes_coarse_to_fine.items():
        fine_labels = [label.strip().replace(' ', '_') for label in fine_labels.split(',')]
        for label in fine_labels:
            assert label not in fine_to_coarse
            fine_to_coarse[label] = coarse_label
    return fine_to_coarse


cifar100_classes_fine_to_coarse = cifar100_fine_to_coarse()


def add_cifar100_superclass(fine_label):
    return f"[{cifar100_classes_fine_to_coarse[fine_label]}]\n{fine_label}"


def get_cifar100_labels(add_coarse_label=False):
    label_names = unpickle_cifar_meta(os.path.join(DATASET_ROOT_DIR, 'cifar100/cifar-100-python/meta'))
    fine_label_names = label_names['fine_label_names']
    if add_coarse_label:
        return [add_cifar100_superclass(label) for label in fine_label_names]
    else:
        return fine_label_names


_TORCHVISION_DATASETS = {
    'mnist':         torchvision.datasets.MNIST,
    'cifar10':       torchvision.datasets.CIFAR10,
    'cifar100':      torchvision.datasets.CIFAR100,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'emnist':        torchvision.datasets.EMNIST
}


_LABEL_NAMES = {
    'cifar10':       ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'cifar100':      get_cifar100_labels(add_coarse_label=False),
    'fashion_mnist': ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
}


def load_mnist(root=None, train=True, transform=None, auto_tensor=True, target_transform=None,
               download=True, return_loader=True, batch_size=16, num_workers=0, shuffle=True):
    """Return a DataLoader for MNIST datasets

    Args:

    - `root`: root dir to cache the downloaded data
    - `train`: if True, use the train set; if False, use the validation set
    - `transform`:
    - `auto_tensor`:
    - `target_transform`:
    - `download`:
    - `return_loader`:
    - `batch_size`:
    - `num_workers`:
    - `shuffle`: Whether shuffle the data, default True

    Return: `DataLoader`
    """
    return _load_torchvision_dataset(
        'mnist', root=root, train=train, transform=transform,
        auto_tensor=auto_tensor, target_transform=target_transform,
        download=download, dataset_kwargs=None, return_loader=return_loader,
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )


def load_emnist(root=None, split='mnist', train=True, transform=None, auto_tensor=True, target_transform=None,
                download=True, return_loader=True, batch_size=16, num_workers=0, shuffle=True):
    return _load_torchvision_dataset(
        'emnist', root=root, train=train, transform=transform,
        auto_tensor=auto_tensor, target_transform=target_transform,
        download=download, dataset_kwargs={'split': split}, return_loader=return_loader,
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )


def load_fashion_mnist(root=None, train=True, transform=None, auto_tensor=True, target_transform=None,
                       download=True, return_loader=True, batch_size=16, num_workers=0, shuffle=True):
    return _load_torchvision_dataset(
        'fashion_mnist', root=root, train=train, transform=transform,
        auto_tensor=auto_tensor, target_transform=target_transform,
        download=download, dataset_kwargs=None, return_loader=return_loader,
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )


def load_cifar10(root=None, train=True, transform=None, auto_tensor=True, target_transform=None,
                 download=True, return_loader=True, batch_size=16, num_workers=0, shuffle=True):
    return _load_torchvision_dataset(
        'cifar10', root=root, train=train, transform=transform,
        auto_tensor=auto_tensor, target_transform=target_transform,
        download=download, dataset_kwargs=None, return_loader=return_loader,
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )


def load_cifar100(root=None, train=True, transform=None, auto_tensor=True,
                  target_transform=None, add_coarse_label=False, download=True,
                  return_loader=True, batch_size=16, num_workers=0, shuffle=True):
    name = 'cifar100'
    if add_coarse_label: _LABEL_NAMES[name] = get_cifar100_labels(add_coarse_label=True)
    return _load_torchvision_dataset(
        name, root=root, train=train, transform=transform,
        auto_tensor=auto_tensor, target_transform=target_transform,
        download=download, dataset_kwargs=None, return_loader=return_loader,
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )


def _load_torchvision_dataset(name, root, train, transform, auto_tensor, target_transform,
                              download, dataset_kwargs, return_loader, **dataloader_kwargs):
    assert name in _TORCHVISION_DATASETS.keys(), f"load_{name} hasn't been supported yet."
    root = os.path.join(DATASET_ROOT_DIR, name) if not root else root
    transform = torchvision.transforms.ToTensor() if (not transform) and auto_tensor else transform
    dataset_cls = _TORCHVISION_DATASETS[name]
    if dataset_kwargs is None:
        dataset = dataset_cls(
            root, train=train, download=download,
            transform=transform,
            target_transform=target_transform
        )
    else:
        assert isinstance(dataset_kwargs, dict)
        dataset = dataset_cls(
            root, train=train, download=download,
            transform=transform,
            target_transform=target_transform,
            **dataset_kwargs
        )
    result = DataLoader(dataset, **dataloader_kwargs) if return_loader else dataset
    if name in _LABEL_NAMES: result.label_names = _LABEL_NAMES[name]
    return result
