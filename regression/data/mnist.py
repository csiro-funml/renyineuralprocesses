import copy
import logging
import os

import torch

from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
def to_numpy(x):
    return x.detach().cpu().numpy()

DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/"))

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

logger = logging.getLogger(__name__)

class MNIST(datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST`.
    """

    shape = (1, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLUE
    name = "MNIST"

    def __init__(
        self, root=DIR_DATA, split="train", subtract_class=[],
            logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        elif split == "val":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            train= split != "test",
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )

        if len(subtract_class)==0: # USUAL SETTING
            # random split training and testing by 8:2
            if split!= 'test':
                x_train, x_val, y_train, y_val = train_test_split(self.data, self.targets, test_size=0.1, shuffle=False)
                self.data = x_train if split == 'train' else x_val
                self.targets =to_numpy(y_train) if split == 'train' else to_numpy(y_val)
            else:
                self.targets = to_numpy(self.targets)

        else: # leave some classes out as validation
            data_train, target_train, data_test, target_test = self._subtract_class(subtract_class=subtract_class)
            # random split training and testing by 8:2
            if split != 'test':
                x_train, x_val, y_train, y_val = train_test_split(data_train, target_train, test_size=0.1)
                self.data = x_train if split == 'train' else x_val
                self.targets = to_numpy(y_train) if split == 'train' else to_numpy(y_val)
            else:
                self.data = data_test
                self.targets = to_numpy(target_test)

    def _subtract_class(self,  subtract_class=[]):
        if len(subtract_class) == 1:
            target = self.targets.clone()
            data = self.data.clone()
            subtract_class = torch.tensor(subtract_class)
            idx = torch.isin(target, subtract_class)

            target_train, target_test = target[~idx], target[idx]
            data_train, data_test = data[~idx], data[idx]
            assert target_train.shape[0] +  target_test.shape[0] == target.shape[0]
        else: # leave the last class test
            target = self.targets.clone()
            data = self.data.clone()
            train_class = torch.tensor(subtract_class[:-1])
            test_class = torch.tensor(subtract_class[-1])
            idx_train = torch.isin(target, train_class)
            idx_test = torch.isin(target, test_class)
            target_train, target_test = target[idx_train], target[idx_test]
            data_train, data_test = data[idx_train], data[idx_test]
            # assert target_train.shape[0] + target_test.shape[0] == target.shape[0]
        return data_train, target_train, data_test, target_test
