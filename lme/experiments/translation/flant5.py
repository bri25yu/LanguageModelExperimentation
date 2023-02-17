from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    FlanT5300MModelMixin,
    FlanT5800MModelMixin,
    FlanT53BModelMixin,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationFlanT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationFlanT5300MExperiment(FlanT5300MModelMixin, TranslationFlanT5ExperimentBase):
    pass


class TranslationFlanT5800MExperiment(FlanT5800MModelMixin, TranslationFlanT5ExperimentBase):
    pass


class TranslationFlanT53BExperiment(FlanT53BModelMixin, TranslationFlanT5ExperimentBase):
    pass
