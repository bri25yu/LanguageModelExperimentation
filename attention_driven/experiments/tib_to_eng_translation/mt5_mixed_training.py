from attention_driven.experiments.model_mixins.mt5_model_mixins import (
    MT5Base580MModelMixin,
    MT5Large1_2BModelMixin,
    MT5FP32Base580MModelMixin,
    MT5FP32Large1_2BModelMixin,
)

from attention_driven.experiments.training_pipelines import FinetuneExperimentBase
from attention_driven.experiments.tib_to_eng_translation.mixed_training import TibToEngWithTibMixin, LongContextMixedTrainingMixin
from attention_driven.experiments.tib_to_eng_translation.finetune_mt5 import MT5TibToEngTranslationMixin


class MT5MixedTrainingMixin(TibToEngWithTibMixin):
    MT5_FP16_WARMUP_NUM_STEPS = MT5TibToEngTranslationMixin.MT5_FP16_WARMUP_NUM_STEPS
    get_translation_training_arguments = MT5TibToEngTranslationMixin.get_translation_training_arguments


class FinetuneMT5Base580MMixedTrainingExperiment(MT5Base580MModelMixin, MT5MixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BMixedTrainingExperiment(MT5Large1_2BModelMixin, MT5MixedTrainingMixin, FinetuneExperimentBase):
    pass


class MT5LongContextMixedTrainingMixin(LongContextMixedTrainingMixin):
    MT5_FP16_WARMUP_NUM_STEPS = MT5TibToEngTranslationMixin.MT5_FP16_WARMUP_NUM_STEPS
    get_translation_training_arguments = MT5TibToEngTranslationMixin.get_translation_training_arguments


class FinetuneMT5Base580MLongContextMixedTrainingExperiment(MT5Base580MModelMixin, MT5LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BLongContextMixedTrainingExperiment(MT5Large1_2BModelMixin, MT5LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5FP32Base580MLongContextMixedTrainingExperiment(MT5FP32Base580MModelMixin, LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5FP32Large1_2BLongContextMixedTrainingExperiment(MT5FP32Large1_2BModelMixin, LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass
