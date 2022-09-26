"""
See the prefix tuning paper at https://arxiv.org/abs/2101.00190
"""
from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from attention_driven.modeling.prefix_tuning import (
    PrefixTuningNaiveM2M100ForConditionalGeneration
)
from attention_driven.experiments.baseline import BaselineExperiment


class PrefixTuningNaiveExperimentBase(BaselineExperiment):
    prefix_length: Union[None, int] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        prefix_length = self.prefix_length

        model = PrefixTuningNaiveM2M100ForConditionalGeneration.from_pretrained(
            model_name, prefix_length=prefix_length
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class PrefixTuningNaiveConfig1Experiment(PrefixTuningNaiveExperimentBase):
    prefix_length = 1


class PrefixTuningNaiveConfig2Experiment(PrefixTuningNaiveExperimentBase):
    prefix_length = 2


class PrefixTuningNaiveConfig3Experiment(PrefixTuningNaiveExperimentBase):
    prefix_length = 5


class PrefixTuningNaiveConfig4Experiment(PrefixTuningNaiveExperimentBase):
    prefix_length = 10
