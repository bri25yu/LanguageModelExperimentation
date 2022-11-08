from lme.training_pipelines import ZeroShotExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.training_argument_mixins import NLLBFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin


class ZeroShotNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMixin, ZeroShotExperimentBase):
    pass


class ZeroShotNLLB600MExperiment(NLLB600MModelMixin, ZeroShotNLLBExperimentBase):
    pass


class ZeroShotNLLB1_3BExperiment(NLLB1_3BModelMixin, ZeroShotNLLBExperimentBase):
    pass
