# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from vissl.data.data_helper import get_mean_image


class NpyDataset(Dataset):
    """
    Synthetic dataset class. Mean image is returned always. This dataset
    is used/recommended to use for testing purposes only.

    Args:
        path (string): can be "" [not used]
        split (string): specify split for the dataset.
            Usually train/val/test. Used to read images if
            reading from a folder `path' and retrieve settings for that split
            from the config path [not used]
        dataset_name (string): name of dataset. For information only. [not used]
        data_source (string, Optional): data source ("synthetic") [not used]
    """

    DEFAULT_SIZE = 50_000

    def __init__(
        self, cfg, path: str, split: str, dataset_name: str, data_source="data_filelist"
    ):
        super(NpyDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.data_source = data_source
        self.dataset = np.load(path)
        self._num_samples = max(len(self.dataset), self.DEFAULT_SIZE, cfg.DATA[split].DATA_LIMIT)

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx: int) -> Tuple[Image.Image, bool]:
        """
        Return the image at index 'idx' and whether the load was successful
        """
        x = self.dataset[idx]
        is_success = True
        return x, is_success
