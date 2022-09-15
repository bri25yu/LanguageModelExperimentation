from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from attention_driven.modeling_attention_driven_dropout import (
    AttentionDrivenM2M100ForConditionalGeneration
)
from attention_driven.experiments.baseline import BaselineExperiment


class AttentionDrivenExperimentBase(BaselineExperiment):
    attention_driven_masking_probability: Union[None, float] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        attention_driven_masking_probability = self.attention_driven_masking_probability

        model = AttentionDrivenM2M100ForConditionalGeneration.from_pretrained(
            model_name, attention_driven_masking_probability
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class AttentionDrivenConfig1Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.15


class AttentionDrivenConfig2Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.25


class AttentionDrivenConfig3Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.50
