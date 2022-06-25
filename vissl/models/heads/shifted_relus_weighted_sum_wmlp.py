# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from charset_normalizer import logging

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.heads.mlp import MLP
from vissl.models.heads.shifted_relus_weighted_sum import ShiftedRelusWeightedSum
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("shifted_relus_weighted_sum_wmlp")
class ShiftedRelusWeightedSumWithMLP(nn.Module):
    """
    This module can be used to attach a layer that shifts relus with different scalars, and returns 
    the weighted sum of those applied on the input, also can have multiple linear layers on top.

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        shifts_linspace: List[int],
        in_channels: int,
        dims: List[int],
        use_first_bn=True,
        use_bn: bool = False,
        use_relu: bool = False,
        fixed_mlp_params = False,
        weights_init_type = "ones"
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            shifts_linspace (list): list of ints: start and endpoint, num steps of linspace in weghted sum  
            in_channels (int): number of channels the input has. This information is
                               used to attached the BatchNorm2D layer.
            dims (int): dimensions of the linear layer. Example [8192, 1000] which means
                        attaches `nn.Linear(8192, 1000, bias=True)`
            use_first_bn (bool): whether to use the batchnorm before linear layer, as in linear_eval_mlp head
        """
        super().__init__()

        self.shifted_relus_weighted_sum = ShiftedRelusWeightedSum(model_config, *shifts_linspace, weights_init_type)

        if use_first_bn:
            self.channel_bn = nn.BatchNorm2d(
                in_channels,
                eps=model_config.HEAD.BATCHNORM_EPS,
                momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
            )
        else:
            self.channel_bn = nn.Identity()
        self.clf = MLP(model_config, dims, use_bn=use_bn, use_relu=use_relu)

        if fixed_mlp_params:
            # Load and set params
            print(self.channel_bn.state_dict())
            device = torch.device('cpu')
            mlp_head_state_dict = torch.load("/home/bbea/data/learned_eval_mlp_after_relu.pt", map_location=device)
            logging.info(f"Loaded state dict contains the following paramd: {mlp_head_state_dict.keys()}")

            self.channel_bn.weight.data = mlp_head_state_dict['0.channel_bn.weight'].cpu()
            self.channel_bn.bias.data = mlp_head_state_dict['0.channel_bn.bias'].cpu()
            self.channel_bn.running_mean = mlp_head_state_dict['0.channel_bn.running_mean'].cpu()
            self.channel_bn.running_var = mlp_head_state_dict['0.channel_bn.running_var'].cpu()
            self.channel_bn.num_batches_tracked = mlp_head_state_dict['0.channel_bn.num_batches_tracked'].cpu()
            self.clf.clf[0].weight.data = mlp_head_state_dict['0.clf.clf.0.weight'].cpu()
            self.clf.clf[0].bias.data = mlp_head_state_dict['0.clf.clf.0.bias'].cpu()

            # Freeze params
            self.channel_bn.track_running_stats = False
            for param in self.channel_bn.parameters():
                param.requires_grad = False

            for param in self.clf.parameters():
                param.requires_grad = False

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
            ), "ShiftedRelusWeightedSumWMLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"ShiftedRelusWeightedSumWMLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        out = self.shifted_relus_weighted_sum(batch)

        # MLP
        if len(out.shape) == 2:
            out = out.unsqueeze(2).unsqueeze(3)
        assert len(out.shape) == 4, "Eval MLP head expects 4D tensor input"
        out = self.channel_bn(out)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)

        return out


@register_model_head("shifted_relus_weighted_sum_wmlp_fsdp")
def ShiftedRelusWeightedSum_WMLPFSDP(
    model_config: AttrDict,
):
    wsum = ShiftedRelusWeightedSumWithMLP(
        model_config,
    )
    wsum = fsdp_auto_wrap_bn(wsum)
    return fsdp_wrapper(wsum, **model_config.FSDP_CONFIG)
