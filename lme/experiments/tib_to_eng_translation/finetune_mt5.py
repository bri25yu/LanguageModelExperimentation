from lme.model_mixins.mt5_model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from lme.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin, TibToEngTranslationWithPrefixMixin
from lme.training_pipelines import FinetuneExperimentBase


class MT5TibToEngTranslationMixin(TibToEngTranslationMixin):
    """
    Since the initial loss for our fp16 mt5 is very high, we add 100 warmup steps to let the model
    backpropogate and update our `lm_scale_modifier` weights first before committing to updating
    any other weights (in a large fashion).
    """


class FinetuneMT5Base580MExperiment(MT5Base580MModelMixin, MT5TibToEngTranslationMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BExperiment(MT5Large1_2BModelMixin, MT5TibToEngTranslationMixin, FinetuneExperimentBase):
    pass


class MT5TibToEngTranslationWithPrefixMixin(TibToEngTranslationWithPrefixMixin):
    pass


class FinetuneMT5Base580MWithPrefixExperiment(MT5Base580MModelMixin, MT5TibToEngTranslationWithPrefixMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BWithPrefixExperiment(MT5Large1_2BModelMixin, MT5TibToEngTranslationWithPrefixMixin, FinetuneExperimentBase):
    pass
