from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from attention_driven.modeling_attention_driven_dropout import (
    AttentionDrivenM2M100ForConditionalGeneration
)
from attention_driven.experiments.baseline import BaselineExperiment


class AttentionDrivenExperimentBase(BaselineExperiment):
    attention_dropout: Union[None, float] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        attention_dropout = self.attention_dropout

        model = AttentionDrivenM2M100ForConditionalGeneration.from_pretrained(
            model_name, attention_dropout
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class AttentionDrivenConfig1Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.05


class AttentionDrivenConfig2Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.10


class AttentionDrivenConfig3Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.15
