from typing import Callable

from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import *
from lme.training_argument_mixins import BloomFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationBloomExperimentBase(BloomFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length", pad_to_multiple_of=max_input_length)


class TranslationBloom600MExperiment(Bloom600MModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom1BExperiment(Bloom1BModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom3BExperiment(Bloom3BModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom7BExperiment(Bloom7BModelMixin, TranslationBloomExperimentBase):
    pass
