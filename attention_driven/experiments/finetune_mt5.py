from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


__all__ = ["FinetuneMT5BaseExperiment", "FinetuneMT5LargeExperiment", "FinetuneMT5XLExperiment"]


class FinetuneMT5ExperimentBase(BaselineV2Experiment):
    MODEL_NAME = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        We don't train the tokenizer on Tibetan corpora at the moment, but this is probably something we want to do.

        https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#t5-like-span-masked-language-modeling
        """
        model_name = self.MODEL_NAME

        tokenizer = MT5Tokenizer.from_pretrained(model_name)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        model.config.max_length = max_input_length

        return model


class FinetuneMT5BaseExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
