# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper

@register_model_head("sign_layer")
class SignLayer(nn.Module):
    """
    This module can be used to apply sign function on the input, or a step function with slope.
    """

    def __init__(
        self,
        model_config: AttrDict,
        slope_param: int
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
        """
        super().__init__()
        self.slope_param = slope_param  
        logging.info(f"Sign alfa value: {self.slope_param}")

    def sign_with_param(self, x):
        if self.slope_param == 0.0:
            return torch.sign(x)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        return torch.max(zeros, torch.min(ones, x / self.slope_param))

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"MLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        out = self.sign_with_param(batch)
        return out


@register_model_head("sign_layer_fsdp")
def SignLayer_FSDP(
    model_config: AttrDict,
):
    sign_layer = SignLayer(
        model_config,
    )
    sign_layer = fsdp_auto_wrap_bn(sign_layer)
    return fsdp_wrapper(sign_layer, **model_config.FSDP_CONFIG)
