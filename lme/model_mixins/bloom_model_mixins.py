from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


__all__ = [
    "Bloom3BModelMixin",
    "Bloom7B1ModelMixin",
]


class BloomModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model


class Bloom3BModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-3b"


class Bloom7B1ModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-7b1"

