from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from attention_driven.modeling.attention_driven_dropout import (
    AttentionDrivenM2M100ForConditionalGeneration
)
from attention_driven.experiments.baseline import BaselineExperiment
from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


class AttentionDrivenExperimentBase(BaselineExperiment):
    attention_dropout: Union[None, float] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        attention_dropout = self.attention_dropout

        model = AttentionDrivenM2M100ForConditionalGeneration.from_pretrained(
            model_name, attention_dropout=attention_dropout
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class AttentionDrivenV2ExperimentBase(BaselineV2Experiment):
    attention_dropout: Union[None, float] = None

    get_model = AttentionDrivenExperimentBase.get_model


class AttentionDrivenConfig1Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.05


class AttentionDrivenConfig2Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.10


class AttentionDrivenConfig3Experiment(AttentionDrivenExperimentBase):
    attention_dropout = 0.15


class AttentionDrivenV2Config1Experiment(AttentionDrivenV2ExperimentBase):
    attention_dropout = 0.10


class AttentionDrivenV2Config2Experiment(AttentionDrivenV2ExperimentBase):
    attention_dropout = 0.10
    MODEL_NAME = "facebook/nllb-200-1.3B"
