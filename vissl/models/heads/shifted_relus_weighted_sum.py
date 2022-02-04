# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("shifted_relus_weighted_sum")
class ShiftedRelusWeightedSum(nn.Module):
    """
    This module can be used to attach a layer that  shifts relus with different scalars, and returns 
    the weighted sum of those applied on the input.

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        linspace_start=0, 
        linspace_end=2, 
        num_steps=10,
        weights_init_type="ones"
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            linspace_start (int): first shift's value
            linspace_end (int): last shift's value
            num_steps (int): number of shifts
            weights_init_type (str): ones/uniform init of weights
        """
        super().__init__()
        err_message = "Last Relu should be removed when using ShiftedRelusWeightedSum layer" 
        assert model_config.TRUNK.RESNETS.REMOVE_LAST_RELU == True, err_message  
        self.weights = nn.Parameter(torch.empty((num_steps),))
        if weights_init_type == "ones":
            nn.init.ones_(self.weights)
        else:
            nn.init.uniform_(self.weights)
        print(f"Shifts: linspace(start: {linspace_start}, end: {linspace_end}, steps: {num_steps})")
        self.shifts = torch.linspace(linspace_start, linspace_end, steps=num_steps)
 
    def calc_weighted_sum(self, batch: torch.Tensor):   
        shifts = self.shifts.to(batch.get_device())
        weighted_sum = torch.zeros_like(batch)
        for i in range(len(shifts)):
            shifted_relu = torch.maximum(torch.zeros_like(batch), batch - shifts[i])
            weighted_sum += torch.abs(self.weights[i]) * shifted_relu
        return weighted_sum

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
            ), "ShiftedRelusWeightedSum input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"ShiftedRelusWeightedSum expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        out = self.calc_weighted_sum(batch)
        return out


@register_model_head("shifted_relus_weighted_sum_fsdp")
def ShiftedRelusWeightedSum_FSDP(
    model_config: AttrDict,
):
    wsum = ShiftedRelusWeightedSum(
        model_config,
    )
    wsum = fsdp_auto_wrap_bn(wsum)
    return fsdp_wrapper(wsum, **model_config.FSDP_CONFIG)
