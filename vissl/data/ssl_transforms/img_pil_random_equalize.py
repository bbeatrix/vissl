# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.data.ssl_transforms.pil_photometric_transforms_lib import (
    EqualizeTransform,
)


@register_transform("ImgPilRandomEqualize")
class ImgPilRandomEqualize(ClassyTransform):
    """
    Randomly apply equalization transform to an image.
    This was used in PAWS - https://arxiv.org/abs/2104.13963
    """

    def __init__(self, prob: float):
        """
        Args:
            p (float): Probability of applying the transform
        """
        self.p = prob
        transforms = [EqualizeTransform()]
        self.transform = pth_transforms.RandomApply(transforms, self.p)
        logging.info(f"ImgPilRandomEqualize with prob {self.p} and {transforms}")

    def __call__(self, image):
        return self.transform(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilRandomEqualize":
        """
        Instantiates ImgPilRandomEqualize from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilRandomEqualize instance.
        """
        prob = config.get("p", 0.2)
        assert isinstance(prob, float), f"p must be a float value. Found {type(prob)}"
        assert prob >= 0 and prob <= 1
        return cls(prob=prob)
