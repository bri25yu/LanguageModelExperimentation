from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.training_argument_mixins import DecayLRFinetuneTrainingArgumentsMixin
from lme.experiments.translation_mixins import TranslationMixin


class FinetuneNLLBExperimentBase(DecayLRFinetuneTrainingArgumentsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class FinetuneNLLB600MExperiment(NLLB600MModelMixin, FinetuneNLLBExperimentBase):
    pass


class FinetuneNLLB1_3BExperiment(NLLB1_3BModelMixin, FinetuneNLLBExperimentBase):
    pass
