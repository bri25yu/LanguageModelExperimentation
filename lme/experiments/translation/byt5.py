from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    ByT5600MModelMixin,
    ByT51BModelMixin,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationByT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationByT5600MExperiment(ByT5600MModelMixin, TranslationByT5ExperimentBase):
    pass


class TranslationByT51BExperiment(ByT51BModelMixin, TranslationByT5ExperimentBase):
    pass
