from typing import Callable

from transformers import default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import *
from lme.training_argument_mixins import BloomFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationBloomExperimentBase(BloomFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return default_data_collator


class TranslationBloom600MExperiment(Bloom600MModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom1BExperiment(Bloom1BModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom3BExperiment(Bloom3BModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom7BExperiment(Bloom7BModelMixin, TranslationBloomExperimentBase):
    pass
