from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, MT5ForConditionalGeneration


__all__ = [
    "MT5600MModelMixin",
    "MT51BModelMixin",
    "MT53BModelMixin",
    "MT513BModelMixin",
]


class MT5ModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model


class MT5600MModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-base"


class MT51BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-large"


class MT53BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-xl"


class MT513BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-xxl"
