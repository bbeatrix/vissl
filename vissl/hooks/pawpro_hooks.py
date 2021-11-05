# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import logging
import torch
import torch.nn as nn
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook


class InitPawPrototypesHook(ClassyHook):
    """
    Initialize prototypes as embedding cluster means in paw training. Optional.
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Initialize prototypes as embedding cluster centroids.
        """
        if not (task.config["LOSS"]["name"] in ["pawpro_loss", "simclr_info_nce_loss"]):
            return
        if not task.config.LOSS[task.config["LOSS"]["name"]].init_protos_with_embs_centroids:
            return
        if not task.train:
            return

        with torch.no_grad():
            try:
                # This is either single GPU model or a FSDP.
                module = getattr(task.model.heads[-1], "prototypes")
                # Determine the context we need to use. For FSDP, we
                # need the summon_full_params context, which ensures that
                # full weights for this layer is all_gathered and after
                # normalization, the changes are persisted in the local
                # shards. All ranks do the same normalization, so all
                # changes should be saved.
                ctx = contextlib.suppress()
                if hasattr(module, "summon_full_params"):
                    ctx = module.summon_full_params()
                with ctx:
                    task.model.heads[-1].init_with_embs_centroids(task)
            except AttributeError:
                # This is a DDP wrapped one.
                task.model.module.heads[-1].init_with_embs_centroids(task)


class ReinitPawPrototypesHook(ClassyHook):
    """
    Reinitialize prototypes as embedding cluster means in paw training. Optional.
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Reinitialize prototypes as embedding cluster centroids.
        """
        if not (task.config["LOSS"]["name"] in ["pawpro_loss", "simclr_info_nce_loss"]):
            return
        if task.config.LOSS[task.config["LOSS"]["name"]].reinit_protos_phase_interval <= 0:
            return
        if not task.train or task.train_phase_idx == 0:
            return
        if task.train_phase_idx % task.config.LOSS[task.config["LOSS"]["name"]].reinit_protos_phase_interval != 0:
            return

        logging.info(f'Current train phase is: {task.train_phase_idx}, reinitializing Paw Prototypes head...')

        with torch.no_grad():
            try:
                # This is either single GPU model or a FSDP.
                module = getattr(task.model.heads[-1], "prototypes")
                # Determine the context we need to use. For FSDP, we
                # need the summon_full_params context, which ensures that
                # full weights for this layer is all_gathered and after
                # normalization, the changes are persisted in the local
                # shards. All ranks do the same normalization, so all
                # changes should be saved.
                ctx = contextlib.suppress()
                if hasattr(module, "summon_full_params"):
                    ctx = module.summon_full_params()
                with ctx:
                    task.model.heads[-1].init_with_embs_centroids(task)
            except AttributeError:
                # This is a DDP wrapped one.
                task.model.module.heads[-1].init_with_embs_centroids(task)
