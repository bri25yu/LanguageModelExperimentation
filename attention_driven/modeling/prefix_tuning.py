from typing import Optional, Tuple

from types import MethodType

import torch
import torch.nn as nn

from transformers import (
    M2M100ForConditionalGeneration, M2M100Config
)


class PrefixTuningM2M100Config(M2M100Config):
    def __init__(self, prefix_length: int=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prefix_length = prefix_length


class PrefixTuningM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    config_class = PrefixTuningM2M100Config

    def __init__(self, config) -> None:
        super().__init__(config)

        self.prefix_length = self.config.prefix_length

        # Freeze all our pretrained model parameters
        for param in self.parameters():
            param.requires_grad = False
