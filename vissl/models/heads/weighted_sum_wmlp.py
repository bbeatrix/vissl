# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.heads.mlp import MLP
from vissl.models.heads.weighted_sum import WeightedSum
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("weighted_sum_wmlp")
class WeightedSumWithMLP(nn.Module):
    """
    This module can be used to attach a layer that shifts input with different scalars, and returns 
    the weighted sum of those.

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_channels: int,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            in_channels (int): number of channels the input has. This information is
                               used to attached the BatchNorm2D layer.
            dims (int): dimensions of the linear layer. Example [8192, 1000] which means
                        attaches `nn.Linear(8192, 1000, bias=True)`
        """
        super().__init__()

        self.weighted_sum = WeightedSum(model_config)
        self.channel_bn = nn.BatchNorm2d(
            in_channels,
            eps=model_config.HEAD.BATCHNORM_EPS,
            momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
        )
        self.clf = MLP(model_config, dims, use_bn=use_bn, use_relu=use_relu)

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

        out = self.weighted_sum(batch)

        # MLP
        if len(out.shape) == 2:
            out = out.unsqueeze(2).unsqueeze(3)
        assert len(out.shape) == 4, "Eval MLP head expects 4D tensor input"
        out = self.channel_bn(out)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)

        return out


@register_model_head("weighted_sum_wmlp_fsdp")
def WeightedSum_WMLPFSDP(
    model_config: AttrDict,
):
    wsum = WeightedSumWMLP(
        model_config,
    )
    wsum = fsdp_auto_wrap_bn(wsum)
    return fsdp_wrapper(wsum, **model_config.FSDP_CONFIG)
