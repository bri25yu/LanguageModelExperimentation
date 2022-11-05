from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer

from attention_driven.modeling.mt5_fp16_utils import scale_weights_for_fp16_t5
from attention_driven.modeling.mt5_fp16 import MT5Fp16ForConditionalGeneration


__all__ = [
    "MT5Base580MModelMixin",
    "MT5Large1_2BModelMixin",
]


class MT5ModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        # We don't have access to bf16 capable Ampere + GPUs so we need to workaround it
        model = MT5Fp16ForConditionalGeneration.from_pretrained(model_name)
        scale_weights_for_fp16_t5(model)

        model.config.max_length = max_input_length

        return model


class MT5Base580MModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-base"


class MT5Large1_2BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-large"
