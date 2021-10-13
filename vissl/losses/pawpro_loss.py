# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint

import numpy as np
import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils. distributed_utils import AllReduce


@register_loss("pawpro_loss")
class PawProLoss(ClassyLoss):
    """
    PAW prototypes loss.

    Config params:
        world_size (int): total number of trainers in training
        num_protos (int): number of prototypes = number of clusters/classes
        temperature (float): the temperature to be applied on the logits
        sharpening_temperature (float): the temperature used for sharpening
        label_smoothing (float): value of label smoothing 
        me_max (bool): whether to use the me-max regularization
    """

    def __init__(self, loss_config: AttrDict):
        super(PawProLoss, self).__init__()
        self.loss_config = loss_config
        self.world_size = self.loss_config.world_size
        self.num_protos = self.loss_config.num_protos
        self.temperature = self.loss_config.temperature
        self.sharpening_temperature = self.loss_config.sharpening_temperature
        self.label_smoothing = self.loss_config.label_smoothing
        self.me_max = self.loss_config.me_max
        self.criterion = PawProCriterion(self.world_size,
                                         self.num_protos,
                                         self.temperature,
                                         self.sharpening_temperature,
                                         self.label_smoothing, 
                                         self.me_max)

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates PawProLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            PawProLoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        normalized_output = nn.functional.normalize(output, dim=1, p=2)
        loss = self.criterion(normalized_output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), 
                     "num_protos": self.num_protos,
                     "temperature": self.temperature,
                     "sharpening_temperature": self.sharpening_temperature,
                     "label_smoothing": self.label_smoothing, 
                     "me_max": self.me_max}
        return pprint.pformat(repr_dict, indent=2)


class PawProCriterion(nn.Module):
    """
    The criterion corresponding to the PAWS loss as defined in the paper
    https://arxiv.org/abs/2104.13963.

    Args:
        world_size (int): total number of trainers in training
        num_protos (int): number of prototypes = number of clusters/classes
        temperature (float): the temperature to be applied on the logits
        sharpening_temperature (float): the temperature used for sharpening
        label_smoothing (float): value of label smoothing 
        me_max (bool): whether to use the me-max regularization
    """

    def __init__(self, world_size: int, num_protos: int, temperature: float,  
                 sharpening_temperature: float, label_smoothing: float, me_max: bool):
        super(PawProCriterion, self).__init__()

        self.world_size = world_size
        self.num_protos = num_protos
        self.temperature = temperature
        self.sharpening_temperature = sharpening_temperature
        self.label_smoothing = label_smoothing
        self.me_max = me_max

        self.num_views = 2
        self.softmax = nn.Softmax(dim=1)
        self.labels = self.create_labels()

    def sharpen(self, p):
        sharp_p = p**(1./self.sharpening_temperature)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def create_labels(self):
        device = torch.cuda.current_device()
        total_protos = self.num_protos * self.world_size

        off_value = self.label_smoothing / self.num_protos

        labels = torch.zeros(total_protos, total_protos).to(device) + off_value
        for i in range(self.num_protos):
            labels[i::self.num_protos][:, i] = 1. - self.label_smoothing + off_value

        return labels

    def forward(self, normalized_embeddings: torch.Tensor):
        batch_size = len(normalized_embeddings) // self.num_views

        anchors = normalized_embeddings
        p_views = torch.cat([anchors[batch_size:], anchors[:batch_size]], dim=0)

        # Step 1: compute anchor predictions
        probs = self.softmax(anchors / self.temperature) @ self.labels

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.softmax(p_views / self.temperature) @ self.labels
            targets = self.sharpen(targets)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        regloss = 0.
        if self.me_max:
            avg_probs = AllReduce.apply(torch.mean(self.sharpen(probs), dim=0))
            regloss -= torch.sum(torch.log(avg_probs**(-avg_probs))) 
            loss += regloss

        assert not torch.isnan(loss).any(), 'pawpro loss is nan oh no'
        assert not torch.isnan(regloss).any(), 'pawpro regloss is nan oh no'
        return loss


    def __repr__(self):
        repr_dict = {
            "name": self._get_name(), 
            "num_protos": self.num_protos,
            "temperature": self.temperature,
            "sharpening_temperature": self.sharpening_temperature,
            "label_smoothing": self.label_smoothing,
            "me_max": self.me_max,
            "num_views": self.num_views
        }
        return pprint.pformat(repr_dict, indent=2)
