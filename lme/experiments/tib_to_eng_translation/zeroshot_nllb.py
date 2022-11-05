from lme.model_mixins.nllb_model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin
from lme.training_pipelines import ZeroShotExperimentBase


class ZeroshotNLLB600MExperiment(NLLB600MModelMixin, TibToEngTranslationMixin, ZeroShotExperimentBase):
    pass


class ZeroshotNLLB1_3BExperiment(NLLB1_3BModelMixin, TibToEngTranslationMixin, ZeroShotExperimentBase):
    pass
