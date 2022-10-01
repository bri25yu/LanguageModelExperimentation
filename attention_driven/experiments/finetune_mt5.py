from collections import OrderedDict

import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, MT5ForConditionalGeneration

from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


__all__ = ["FinetuneMT5BaseExperiment", "FinetuneMT5LargeExperiment", "FinetuneMT5XLExperiment"]


class FinetuneMT5ExperimentBase(BaselineV2Experiment):
    MODEL_NAME = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        We don't train the tokenizer on Tibetan corpora at the moment, but this is probably something we want to do.

        https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#t5-like-span-masked-language-modeling
        """

        tokenizer = AutoTokenizer.from_pretrained("buddhist-nlp/mt5-tibetan-tokenizer")

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        # Load pretrained parameter weights
        base_model_parameter_dict = AutoModelForSeq2SeqLM.from_pretrained(model_name).state_dict()
        base_model_parameter_dict = OrderedDict(base_model_parameter_dict)  # Make `base_model_parameter_dict` modifiable
        pretrained_embedding_weight = base_model_parameter_dict.pop("shared.weight")

        # Create new model
        config = AutoConfig.from_pretrained(model_name, vocab_size=tokenizer.vocab_size + 2)
        model = MT5ForConditionalGeneration(config)

        # Load pretrained weights into new model with a slight change to embeddings
        # since we have a larger vocab size
        model.load_state_dict(base_model_parameter_dict)
        pretrained_vocab_size, hidden_dim = pretrained_embedding_weight.size()
        with torch.no_grad():
            model.shared.weight[:pretrained_vocab_size, :hidden_dim].copy_(pretrained_embedding_weight)

        model.config.max_length = max_input_length

        return model


class FinetuneMT5BaseExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
