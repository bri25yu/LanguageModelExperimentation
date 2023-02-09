from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1BModelMixin, NLLB3BModelMixin
from lme.training_argument_mixins import NLLBFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationNLLB600MExperiment(NLLB600MModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB1BExperiment(NLLB1BModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB3BExperiment(NLLB3BModelMixin, TranslationNLLBExperimentBase):
    pass 
