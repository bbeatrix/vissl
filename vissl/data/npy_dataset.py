from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from vissl.data.data_helper import get_mean_image

import logging

from iopath.common.file_io import g_pathmgr
from torchvision.datasets import ImageFolder
from vissl.data.data_helper import QueueDataset, get_mean_image
from vissl.utils.io import load_file

class NpyDataset(QueueDataset):
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
        self, cfg, path: str, split: str, dataset_name: str, data_source="disk_filelist"
    ):
        super(NpyDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        self.cfg = cfg
        self.split = split
        self.data_source = data_source
        #   self.dataset = np.load(path)
        #   self._num_samples = max(len(self.dataset), self.DEFAULT_SIZE, cfg.DATA[split].DATA_LIMIT)

        self.dataset_name = dataset_name
        self._path = path
        self.npy_dataset = []
        self.is_initialized = False
        self._load_data(path)
        self._num_samples = len(self.npy_dataset)

        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

    def _load_data(self, path):
        if self.cfg["DATA"][self.split].MMAP_MODE:
            self.npy_dataset = load_file(path, mmap_mode="r")
        else:
            self.npy_dataset = load_file(path)

        logging.info(f"Loaded {len(self.npy_dataset)} samples from {path}")
        self.is_initialized = True


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

    def __getitem__(self, idx: int) -> Tuple[np.array, bool]:
        """
        Return the image at index 'idx' and whether the load was successful
        """
        if not self.is_initialized:
            self._load_data(self._path)
            self.is_initialized = True
        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()
        is_success = True
        x = self.npy_dataset[idx]

        return x, is_success
