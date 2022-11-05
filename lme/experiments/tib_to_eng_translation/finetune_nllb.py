from lme.model_mixins.nllb_model_mixins import NLLB600MModelMixin, NLLB1_3BModelMixin
from lme.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin
from lme.training_pipelines import FinetuneExperimentBase


class FinetuneNLLB600MExperiment(NLLB600MModelMixin, TibToEngTranslationMixin, FinetuneExperimentBase):
    pass


class FinetuneNLLB1_3BExperiment(NLLB1_3BModelMixin, TibToEngTranslationMixin, FinetuneExperimentBase):
    pass
