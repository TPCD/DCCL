from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


import os.path as osp
from torch.utils.data import DataLoader, Dataset

from PIL import Image


class FakeLabelDataset(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        if isinstance(self.dataset[0][0], str):
            self.data0_is_numpy = False
        else:
            self.data0_is_numpy = True
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.data0_is_numpy:
            image_numpy, fake, real = self.dataset[index]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real

        else:
            fname, fake, real = self.dataset[index]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real
