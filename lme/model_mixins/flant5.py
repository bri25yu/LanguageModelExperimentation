from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, T5ForConditionalGeneration


__all__ = [
    "FlanT5300MModelMixin",
    "FlanT5800MModelMixin",
    "FlanT53BModelMixin",
]


class FlanT5ModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model


class FlanT5300MModelMixin(FlanT5ModelMixinBase):
    MODEL_NAME = "google/flan-t5-base"


class FlanT5800MModelMixin(FlanT5ModelMixinBase):
    MODEL_NAME = "google/flan-t5-large"


class FlanT53BModelMixin(FlanT5ModelMixinBase):
    MODEL_NAME = "google/flan-t5-xl"
