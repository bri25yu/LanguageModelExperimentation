from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MT5600MModelMixin,
    MT51BModelMixin,
    MT53BModelMixin,
    MT513BModelMixin,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin, MT513BFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationMT5600MExperiment(MT5600MModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT51BExperiment(MT51BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT53BExperiment(MT53BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT513BExperiment(MT513BModelMixin, MT513BFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass
