"""
Adapted from https://github.com/huggingface/transformers/issues/14189#issuecomment-961571628

The following weights are not scaled (but maybe they should be):
    'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
    'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
    'lm_head.weight'

"""
from typing import Union

import torch

from transformers import T5PreTrainedModel


__all__ = ["scale_weights_for_fp16_t5"]


# The same architectures in the model share the same weight scaling
# The embeddings scaling is the maximum scaling performed
ARCH_SCALING = {
    "embeddings": 1 / 32.0,
    "attention_value": 1 / 4.0,
    "attention_output": 1 / 8.0,
    "feedforward_weights_in": 1 / 4.0,
    "feedforward_weights_out": 1 / 4.0,
    "feedforward_layernorm": 1 / 2.0,
}

assert ARCH_SCALING["attention_value"] * ARCH_SCALING["attention_output"] == ARCH_SCALING["embeddings"]
assert ARCH_SCALING["feedforward_weights_in"] * ARCH_SCALING["feedforward_weights_out"] * ARCH_SCALING["feedforward_layernorm"] == ARCH_SCALING["embeddings"]


WEIGHT_SCALING = {
    "shared.weight": ARCH_SCALING["embeddings"],
    "SelfAttention.v": ARCH_SCALING["attention_value"],
    "SelfAttention.o": ARCH_SCALING["attention_output"],
    "EncDecAttention.v": ARCH_SCALING["attention_value"],
    "EncDecAttention.o": ARCH_SCALING["attention_output"],
    "lm_scale_modifier": 1 / ARCH_SCALING["embeddings"],
    "DenseReluDense.wi": ARCH_SCALING["feedforward_weights_in"],
    "DenseReluDense.wo": ARCH_SCALING["feedforward_weights_out"],
    "layer_norm": ARCH_SCALING["feedforward_layernorm"],
}


def scale_weights_for_fp16_t5(model: T5PreTrainedModel) -> None:
    assert hasattr(model, "lm_scale_modifier"), "This function is only to be used with a modified mt5 model"

    def search_for_scaling(weight_name: str) -> Union[None, float]:
        for weight_name_infix in WEIGHT_SCALING:
            if weight_name_infix in weight_name:
                return WEIGHT_SCALING[weight_name_infix]

        return None

    with torch.no_grad():
        for weight_name, weight in model.state_dict().items():
            scaling = search_for_scaling(weight_name)
            if scaling is None:
                continue

            weight *= scaling
