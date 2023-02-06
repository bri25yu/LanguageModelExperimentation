from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MT5Base580MModelMixin,
    MT5Large1_2BModelMixin,
    MT53BModelMixin,
    MT513BModelMixin,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationMT5BaseExperiment(MT5Base580MModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT5LargeExperiment(MT5Large1_2BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT53BExperiment(MT53BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT513BExperiment(MT513BModelMixin, TranslationMT5ExperimentBase):
    pass
