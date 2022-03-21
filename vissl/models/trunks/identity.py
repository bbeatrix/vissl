# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

from typing import List
from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk


@register_model_trunk("identity")
class IdentityModel(nn.Module):
    def __init__(self, model_config: AttrDict):
        super(IdentityModel, self).__init__()
        self.identity = nn.Identity()
        self.model_config = model_config
        logging.info("Identity trunk.")

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        # print("\n nn identity forward \n ", x.shape)
        x = x.permute((0, 2, 1, 3))
        # print("\n nn identity forward \n ", x.shape)
        return [self.identity(x)]
