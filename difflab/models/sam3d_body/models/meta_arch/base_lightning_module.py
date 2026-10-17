# Copyright (c) Meta Platforms, Inc. and affiliates.
# difflab: Lightning only trained this; inference needs a Module that knows its device.

import torch.nn as nn


class BaseLightningModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def save_hyperparameters(self, *args, **kwargs):
        pass
