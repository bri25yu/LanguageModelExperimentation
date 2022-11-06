import os

from transformers import Seq2SeqTrainingArguments, TrainingArguments

from lme.model_mixins.mt5_model_mixins import (
    MT5Base580MModelMixin,
    MT5Large1_2BModelMixin,
    MT5FP32Base580MModelMixin,
    MT5FP32Large1_2BModelMixin,
)

from lme.training_pipelines import FinetuneExperimentBase
from lme.experiments.tib_to_eng_translation.mixed_training import (
    TibToEngWithTibMixin,
    LongContextMixedTrainingMixin,
    LC_MT_v2_Mixin,
    LC_MT_v3_Mixin,
)


class MT5MixedTrainingMixin(TibToEngWithTibMixin):
    pass


class FinetuneMT5Base580MMixedTrainingExperiment(MT5Base580MModelMixin, MT5MixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BMixedTrainingExperiment(MT5Large1_2BModelMixin, MT5MixedTrainingMixin, FinetuneExperimentBase):
    pass


class MT5LongContextMixedTrainingMixin(LongContextMixedTrainingMixin):
    pass


class FinetuneMT5Base580MLongContextMixedTrainingExperiment(MT5Base580MModelMixin, MT5LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BLongContextMixedTrainingExperiment(MT5Large1_2BModelMixin, MT5LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5FP32Base580MLongContextMixedTrainingExperiment(MT5FP32Base580MModelMixin, LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5FP32Large1_2BLongContextMixedTrainingExperiment(MT5FP32Large1_2BModelMixin, LongContextMixedTrainingMixin, FinetuneExperimentBase):
    pass


class MT5_LC_MT_v2_Mixin(LC_MT_v2_Mixin):
    pass


class Finetune_MT5Base_LC_MT_v2Experiment(MT5Base580MModelMixin, MT5_LC_MT_v2_Mixin, FinetuneExperimentBase):
    pass


class Finetune_MT5Large_LC_MT_v2Experiment(MT5Large1_2BModelMixin, MT5_LC_MT_v2_Mixin, FinetuneExperimentBase):
    pass


class MT5_LC_MT_SLS_v2_Mixin(MT5_LC_MT_v2_Mixin):
    """
    LC_MT_SLS stands for long context mixed training static loss scale
    """
    pass


class Finetune_MT5Base_LC_MT_SLS_v2Experiment(MT5Base580MModelMixin, MT5_LC_MT_SLS_v2_Mixin, FinetuneExperimentBase):
    pass


class Finetune_MT5Large_LC_MT_SLS_v2Experiment(MT5Large1_2BModelMixin, MT5_LC_MT_SLS_v2_Mixin, FinetuneExperimentBase):
    pass


class MT5_LC_MT_SLS_v3_Mixin(LC_MT_v3_Mixin):
    pass


class Finetune_MT5Base_LC_MT_SLS_v3Experiment(MT5Base580MModelMixin, MT5_LC_MT_SLS_v3_Mixin, FinetuneExperimentBase):
    pass


class Finetune_MT5Large_LC_MT_SLS_v3Experiment(MT5Large1_2BModelMixin, MT5_LC_MT_SLS_v3_Mixin, FinetuneExperimentBase):
    pass
