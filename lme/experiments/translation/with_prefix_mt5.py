from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.translation.with_prefix_mixin import TranslationWithPrefixMixin


class TranslationWithPrefixMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationWithPrefixMixin, FinetuneExperimentBase):
    pass


class TranslationWithPrefixMT5BaseExperiment(MT5Base580MModelMixin, TranslationWithPrefixMT5ExperimentBase):
    pass


class TranslationWithPrefixMT5LargeExperiment(MT5Large1_2BModelMixin, TranslationWithPrefixMT5ExperimentBase):
    pass
