# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("weighted_sum")
class WeightedSum(nn.Module):
    """
    This module can be used to attach a layer that shifts input with different scalars, and returns 
    the weighted sum of those.

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty((10),))
        nn.init.uniform_(self.weights)
        self.weights.data = self.weights / torch.sum(self.weights)
        self.shifts = torch.linspace(0, 2, steps=10)
        
    def calc_weighted_sum(self, batch: torch.Tensor):
        x = torch.unsqueeze(batch, 0)
        shifts = self.shifts.to(self.weights.get_device())
        repeated_x = x.repeat((self.shifts.shape[0], 1, 1))
        shifted_x =  x + torch.einsum('i, ijk -> ijk', shifts, repeated_x)
        weighted_shifted_x = torch.einsum('i, ijk -> ijk', self.weights, shifted_x)
        weighted_sum = torch.sum(weighted_shifted_x, dim=0, keepdims=True)
        return torch.squeeze(weighted_sum)

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
            ), "WeightedSumWMLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"WeightedSumWMLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        out = self.calc_weighted_sum(batch)
        return out


@register_model_head("weighted_sum_fsdp")
def WeightedSum_FSDP(
    model_config: AttrDict,
):
    wsum = WeightedSum(
        model_config,
    )
    wsum = fsdp_auto_wrap_bn(wsum)
    return fsdp_wrapper(wsum, **model_config.FSDP_CONFIG)
