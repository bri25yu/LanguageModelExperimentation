import os

from transformers import Seq2SeqTrainingArguments, TrainingArguments

from attention_driven.experiments.model_mixins.mt5_model_mixins import (
    MT5Base580MModelMixin,
    MT5Large1_2BModelMixin,
    MT5FP32Base580MModelMixin,
    MT5FP32Large1_2BModelMixin,
)

from attention_driven.experiments.training_pipelines import FinetuneExperimentBase
from attention_driven.experiments.tib_to_eng_translation.mixed_training import (
    TibToEngWithTibMixin,
    LongContextMixedTrainingMixin,
    LC_MT_v2_Mixin,
)
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


class MT5_LC_MT_v2_Mixin(LC_MT_v2_Mixin):
    MT5_FP16_WARMUP_NUM_STEPS = MT5TibToEngTranslationMixin.MT5_FP16_WARMUP_NUM_STEPS

    # This is an exact copy of `MT5TibToEngTranslationMixin.get_translation_training_arguments` unless specified otherwise
    def get_translation_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )
        max_steps = self.NUM_TRANSLATION_TRAIN_STEPS
        eval_steps = self.NUM_TRANSLATION_EVAL_STEPS
        target_total_batch_size_per_update = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE
        world_size = self.get_world_size()
        per_gpu_batch_size = batch_size

        eval_save_strategy = "steps"

        gradient_accumulation_steps = target_total_batch_size_per_update // (per_gpu_batch_size * world_size)
        gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

        return Seq2SeqTrainingArguments(
            output_dir,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            evaluation_strategy=eval_save_strategy,
            save_strategy=eval_save_strategy,
            max_steps=max_steps,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=1,
            per_device_train_batch_size=per_gpu_batch_size,
            per_device_eval_batch_size=per_gpu_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=1,
            do_train=True,
            do_eval=True,
            seed=42,
            fp16=True,
            log_level="error",
            log_on_each_node=False,
            logging_steps=1,
            predict_with_generate=True,
            warmup_steps=self.MT5_FP16_WARMUP_NUM_STEPS,

            ###########################
            # START use warmup and decay lr schedule
            ###########################

            # Original code:
            # deepspeed=self.load_deepspeed_template_args("WarmupLR"),

            deepspeed=self.load_deepspeed_template_args("WarmupDecayLR"),

            ###########################
            # END use warmup and decay lr schedule
            ###########################
        )


class Finetune_MT5Base_LC_MT_v2Experiment(MT5Base580MModelMixin, MT5_LC_MT_v2_Mixin, FinetuneExperimentBase):
    pass


class Finetune_MT5Large_LC_MT_v2Experiment(MT5Large1_2BModelMixin, MT5_LC_MT_v2_Mixin, FinetuneExperimentBase):
    pass
