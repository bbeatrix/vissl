# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

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
        if not task.config["LOSS"]["name"] == "pawpro_loss":
            return
        if not task.config.LOSS["pawpro_loss"].init_protos_with_embs_centroids:
            return

        phase_type = "train" if task.train else "test"

        with torch.no_grad():
            try:
                # This is either single GPU model or a FSDP.
                for j in range(task.model.heads[-1].num_heads):
                    module = getattr(task.model.heads[-1], "prototypes" + str(j))
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
