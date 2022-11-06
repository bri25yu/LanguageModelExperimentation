from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.training_argument_mixins import DecayLRFinetuneTrainingArgumentsMixin
from lme.experiments.translation_mixins import TranslationMixin


class TranslationNLLBExperimentBase(DecayLRFinetuneTrainingArgumentsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationNLLB600MExperiment(NLLB600MModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB1_3BExperiment(NLLB1_3BModelMixin, TranslationNLLBExperimentBase):
    pass
