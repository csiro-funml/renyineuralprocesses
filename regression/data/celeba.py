import copy

import torch
import random
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
from utils.paths import datasets_path

DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/"))

class CelebA(object):
    def __init__(self, split="train"):
        self.data, self.targets = torch.load(
                osp.join(datasets_path, 'celeba',
                    'train.pt' if split=="train" else 'eval.pt'))
        self.data = self.data.float() / 255.0

        if split=="train":
            self.data, self.targets = self.data, self.targets
        else:
            self.data, self.targets = self.data, self.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


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

def preprocess(root, size=(64, 64), img_format="JPEG", center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, "*" + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


class ExternalDataset(Dataset, abc.ABC):
    """Base Class for external datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.dir = os.path.join(root, self.name)
        self.train_data = os.path.join(self.dir, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(self.dir):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class CelebA64(ExternalDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """

    urls = {
        "train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
    }
    files = {"train": "img_align_celeba"}
    shape = (3, 64, 64)
    missing_px_color = COLOUR_BLACK
    n_classes = 0  # not classification
    name = "celeba64"

    def __init__(self, split='train', root=DIR_DATA, truncate_dataset=False, extract_class=[], **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        imgs = glob.glob(self.train_data + "/*")
        if len(extract_class) == 0:
            # random split training and testing by 8:2
            x_train_val, x_test = train_test_split(imgs, test_size=0.2)
            x_train, x_val = train_test_split(x_train_val, test_size=0.1)
            print("training shape:", len(x_train), "validating shape", len(x_val), "testing shape:", len(x_test))
            if split == 'train':
                if truncate_dataset == True:
                    MAX_TRAINING = 10000
                    # print(x_train[:10])
                    random.shuffle(x_train)
                    self.imgs = x_train[:MAX_TRAINING]
                    print("training shape", len(self.imgs))
                    print(self.imgs[:10])
                else:
                    self.imgs = x_train
            elif split=='val':
                self.imgs = x_val
            else:
                self.imgs = x_test
                if truncate_dataset == True:
                    MAX_TRAINING = 1000
                    self.imgs = x_test[:MAX_TRAINING]

        else:
            imgs_train, imgs_test = self._extract_by_target(imgs, extract_class=extract_class)

            # random split training and testing by 8:2
            # x_train_val, x_test = train_test_split(imgs_train, test_size=0.2)
            x_train, x_val = train_test_split(imgs_train, test_size=0.2)
            x_test = imgs_test
            print("training shape:", len(x_train), "validating shape", len(x_val), "testing shape:", len(x_test))
            if split == 'train':
                if truncate_dataset == True:
                    MAX_TRAINING = 10000
                    # print(x_train[:10])
                    random.shuffle(x_train)
                    self.imgs = x_train[:MAX_TRAINING]
                    print("training shape", len(self.imgs))
                    print(self.imgs[:10])
                else:
                    self.imgs = x_train
            elif split == 'val':
                self.imgs = x_val
            else:
                self.imgs = x_test
                if truncate_dataset == True:
                    MAX_TRAINING = 1000
                    self.imgs = x_test[:MAX_TRAINING]
    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.dir, "celeba.zip")
        os.makedirs(self.dir)

        try:
            subprocess.check_call(
                ["curl", "-L", type(self).urls["train"], "--output", save_path]
            )
        except FileNotFoundError as e:
            raise Exception(e + " Please instal curl with `apt-get install curl`...")

        hash_code = "00d2c5bc6d35e252742224ab0c1e8fcb"
        assert (
            hashlib.md5(open(save_path, "rb").read()).hexdigest() == hash_code
        ), "{} file is corrupted.  Remove the file and try again.".format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.dir)

        os.remove(save_path)

        self.preprocess()

    def preprocess(self):
        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = plt.imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


    def _extract_by_target(self, imgs, extract_class=[]):
        label_values = torch.unique(self.target[extract_class[0]])
        print("unique labels")

        idx = self.target[extract_class[0] == label_values[0]]
        imgs_train, imgs_test = imgs[idx], imgs[~idx]
        return imgs_train, imgs_test

class CelebA32(CelebA64):
    shape = (3, 32, 32)
    name = "celeba32"



if __name__ == '__main__':
    import os
    import os.path as osp
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import torch

    # load train/val/test split
    splitdict = {}
    with open(osp.join(datasets_path, 'celeba', 'list_eval_partition.txt'), 'r') as f:
        for line in f:
            fn, split = line.split()
            splitdict[fn] = int(split)

    # load identities
    iddict = {}
    with open(osp.join(datasets_path, 'celeba', 'identity_CelebA.txt'), 'r') as f:
        for line in f:
            fn, label = line.split()
            iddict[fn] = int(label)

    train_imgs = []
    train_labels = []
    eval_imgs = []
    eval_labels = []
    path = osp.join(datasets_path, 'celeba', 'img_align_celeba')
    imgfilenames = os.listdir(path)
    for fn in tqdm(imgfilenames):

        img = Image.open(osp.join(path, fn)).resize((32, 32))
        if splitdict[fn] == 2:
            eval_imgs.append(torch.LongTensor(np.array(img).transpose(2, 0, 1)))
            eval_labels.append(iddict[fn])
        else:
            train_imgs.append(torch.LongTensor(np.array(img).transpose(2, 0, 1)))
            train_labels.append(iddict[fn])

    print(f'{len(train_imgs)} train, {len(eval_imgs)} eval')

    train_imgs = torch.stack(train_imgs)
    train_labels = torch.LongTensor(train_labels)
    torch.save([train_imgs, train_labels], osp.join(datasets_path, 'celeba', 'train.pt'))

    eval_imgs = torch.stack(eval_imgs)
    eval_labels = torch.LongTensor(eval_labels)
    torch.save([eval_imgs, eval_labels], osp.join(datasets_path, 'celeba', 'eval.pt'))
