from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, BloomForCausalLM


__all__ = [
    "Bloom600MModelMixin",
    "Bloom1BModelMixin",
    "Bloom3BModelMixin",
    "Bloom7BModelMixin",
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

        model = BloomForCausalLM.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model


class Bloom600MModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-560m"


class Bloom1BModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-1b7"


class Bloom3BModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-3b"


class Bloom7BModelMixin(BloomModelMixinBase):
    MODEL_NAME = "bigscience/bloom-7b1"
