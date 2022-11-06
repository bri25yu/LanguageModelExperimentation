from lme.training_pipelines import ZeroShotExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.training_argument_mixins import DecayLRFinetuneTrainingArgumentsMixin
from lme.experiments.translation_mixins import TranslationMixin


class ZeroShotNLLBExperimentBase(DecayLRFinetuneTrainingArgumentsMixin, TranslationMixin, ZeroShotExperimentBase):
    pass


class ZeroShotNLLB600MExperiment(NLLB600MModelMixin, ZeroShotNLLBExperimentBase):
    pass


class ZeroShotNLLB1_3BExperiment(NLLB1_3BModelMixin, ZeroShotNLLBExperimentBase):
    pass
