from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin, NLLB3_3BModelMixin
from lme.training_argument_mixins import NLLBFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationNLLB600MExperiment(NLLB600MModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB1_3BExperiment(NLLB1_3BModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB3_3BExperiment(NLLB3_3BModelMixin, TranslationNLLBExperimentBase):
    pass 
