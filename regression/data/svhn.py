import torch
import os.path as osp
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import abc
import glob
import hashlib
import logging
import os
import subprocess
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = torch.tensor([0.0, 0.0, 0.0])
COLOUR_WHITE = torch.tensor([1.0, 1.0, 1.0])
COLOUR_BLUE = torch.tensor([0.0, 0.0, 1.0])
DATASETS_DICT = {
    "mnist": "MNIST",
    "svhn": "svhn",
    "celeba32": "CelebA32",
    "celeba64": "CelebA64",
    "zs-multi-mnist": "ZeroShotMultiMNIST",
    "zsmm": "ZeroShotMultiMNIST",  # shorthand
    "zsmmt": "ZeroShotMultiMNISTtrnslt",
    "zsmms": "ZeroShotMultiMNISTscale",
    "zs-mnist": "ZeroShotMNIST",
    "celeba": "CelebA",
    "celeba128": "CelebA128",
}
DATASETS = list(DATASETS_DICT.keys())

DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/"))

def to_numpy(x):
    return x.detach().cpu().numpy()


class SVHN(datasets.SVHN):
    """svhn wrapper. Docs: `datasets.svhn.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] isntead of [-1,1] to make it easier to use
    probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """

    shape = (3, 32, 32)
    missing_px_color = COLOUR_BLACK
    n_classes = 10
    name = "svhn"

    def __init__(
        self, root=DIR_DATA, split="train", logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [  # transforms.Lambda(lambda x: random_translation(x, 2)),
                transforms.ToTensor()
            ]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        elif split == "val":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            split= split if split == "test" else "train",
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )


        #TODO: for testing purpose only
        if not torch.cuda.is_available():
            n_testing_samples = 200
            self.data = self.data[:n_testing_samples]
            self.labels = self.labels[:n_testing_samples]

        # random split training and testing by 8:2
        if split != 'test':
            x_train, x_val, y_train, y_val = train_test_split(self.data, self.targets, test_size=0.1)
            self.data = x_train if split == 'train' else x_val
            self.labels = y_train if split == 'train' else y_val
        else:
            self.labels = self.labels

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels

    @targets.setter
    def targets(self, value):
        self.labels = value