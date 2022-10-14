"""
See the LoRA paper at https://arxiv.org/abs/2106.09685
"""
from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from attention_driven.modeling.lora import (
    LoRAM2M100ForConditionalGeneration
)
from attention_driven.experiments.baseline import BaselineExperiment


class LoRAExperimentBase(BaselineExperiment):
    rank: Union[None, int] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        rank = self.rank

        model = LoRAM2M100ForConditionalGeneration.from_pretrained(
            model_name, rank=rank
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class LoRAConfig1Experiment(LoRAExperimentBase):
    rank = 1


class LoRAConfig2Experiment(LoRAExperimentBase):
    rank = 2


class LoRAConfig3Experiment(LoRAExperimentBase):
    rank = 4


class LoRAConfig4Experiment(LoRAExperimentBase):
    rank = 8


class LoRAConfig5Experiment(LoRAExperimentBase):
    rank = 64
