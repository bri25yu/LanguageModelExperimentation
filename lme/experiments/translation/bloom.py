from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import Bloom3BModelMixin, Bloom7BModelMixin
from lme.training_argument_mixins import BloomFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationBloomExperimentBase(BloomFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationBloom3BExperiment(Bloom3BModelMixin, TranslationBloomExperimentBase):
    pass


class TranslationBloom7BExperiment(Bloom7BModelMixin, TranslationBloomExperimentBase):
    pass
