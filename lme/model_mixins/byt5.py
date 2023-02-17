from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, T5ForConditionalGeneration


__all__ = [
    "ByT5600MModelMixin",
    "ByT51BModelMixin",
]


class ByT5ModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME

        model = T5ForConditionalGeneration.from_pretrained(model_name)

        return model


class ByT5600MModelMixin(ByT5ModelMixinBase):
    MODEL_NAME = "google/byt5-base"


class ByT51BModelMixin(ByT5ModelMixinBase):
    MODEL_NAME = "google/byt5-large"
