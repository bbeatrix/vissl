# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from vissl.config import AttrDict
from vissl.data import build_dataloader, build_dataset
from vissl.models.heads import register_model_head


def KMeans(x, K=10, n_iter=20, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric.
    :param x Tensor with shape (num_proto, proto_size)
    :param K number of desired clusters
    :param n_iter number of iterations to run
    :param verbose If log speed or not
    @return cl Cluster assignements
    @return c Cluster centroids
    """
    start = time.time()

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone().to(x.device)  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in tqdm(range(n_iter), desc="KMeans clustering"):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                n_iter, end - start, n_iter, (end - start) / n_iter
            )
        )

    return cl, c


@register_model_head("pawpro_head")
class PawPrototypesHead(nn.Module):
    """
    Paw prototypes head (on top of separate projection head).
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        normalize_feats: bool = True,
        use_weight_norm_prototypes: bool = False
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            dims (int): dimensions of the linear layer. Must have length 2.
            normalize_feats (bool): whether to normalize inputs of prototype head.
            use_weight_norm_prototypes (bool): whether to use weight norm module for the prototypes layers.
        """

        super().__init__()
        self.normalize_feats = normalize_feats
        assert len(dims) == 2, "dims list of prototype layer must have lenght of 2."
        self.num_protos = dims[-1]

        # prototypes (i.e. centroids) layer
        protos_layer = nn.Linear(dims[0], self.num_protos, bias=False)
        if use_weight_norm_prototypes:
            protos_layer = nn.utils.weight_norm(protos_layer)
            protos_layer.weight_g.data.fill_(1)
        self.add_module("prototypes", protos_layer)

    def init_with_embs_centroids(self, task):
        logging.info("Init Paw Prototypes head...")
        if not task.train:
            logging.info("Paw Prototypes head can be initialized only during training phases, skipping it now")
            return

        init_dataloader = self._create_init_dataloader(task)
        init_data_iterator = iter(init_dataloader)

        zs = []
        z_num = 0
        for itr, batch in enumerate(init_data_iterator):
            assert len(batch['data']) == 1, "Init batch data list should contain only a single tensor"
            img_batch = batch['data'][0]

            with torch.no_grad():
                try:
                    z = task.model.forward_to_last_head(img_batch)[0]
                except AttributeError:
                    z = task.model.module.forward_to_last_head(img_batch)[0]
                zs.append(z.cpu())
                z_num += len(z)
        embeddings = torch.cat(zs)
        embeddings = nn.functional.normalize(embeddings, dim=-1)

        with torch.no_grad():
            cl, means = KMeans(embeddings, self.num_protos, n_iter=10, verbose=True)
            getattr(self, "prototypes").weight.data.copy_(means)
        logging.info("Init finished successfully!")

    def _create_init_dataloader(self, task):
        phase_type = "TRAIN"

        task_device = task.device
        task_config = task.config
        data_config = {'DATA': task_config['DATA']}
        data_config['DATA']['TRAIN']["TRANSFORMS"] = [
            {'name': 'Resize', 'size': 36},
            {'name': 'CenterCrop', 'size': 32},
            {'name': 'RandomHorizontalFlip', 'p': 0.5},
            {'name': 'ToTensor'},
            {'name': 'Normalize', 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]}
        ]
        data_config['DATA']['TRAIN']['COLLATE_FUNCTION'] = 'default_collate'
        data_config['DATA']['TRAIN']['COLLATE_FUNCTION_PARAMS'] = {}

        init_dataset = build_dataset(data_config, phase_type)

        init_dataloader = build_dataloader(
            dataset=init_dataset,
            dataset_config=data_config['DATA'][phase_type],
            num_dataloader_workers=task_config.DATA.NUM_DATALOADER_WORKERS,
            pin_memory=task_config.DATA.PIN_MEMORY,
            multi_processing_method=task_config.MULTI_PROCESSING_METHOD,
            device=task_device,
            sampler_seed=task_config["SEED_VALUE"],
            split=phase_type.lower(),
        )
        return init_dataloader

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (4D torch.tensor): shape (N x C x H x W)
        Returns:
            List(2D torch.tensor of shape N x num_clusters)
        """

        if self.normalize_feats:
            batch = nn.functional.normalize(batch, dim=1, p=2)

        out = getattr(self, "prototypes")(batch)
        return out
