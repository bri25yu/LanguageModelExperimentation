import os

from transformers import TrainingArguments, SchedulerType, Seq2SeqTrainingArguments

from attention_driven.experiments.model_mixins.mt5_model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from attention_driven.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin
from attention_driven.experiments.training_pipelines import FinetuneExperimentBase


class MT5TibToEngTranslationMixin(TibToEngTranslationMixin):
    """
    Since the initial loss for our fp16 mt5 is very high, we add 100 warmup steps to let the model
    backpropogate and update our `lm_scale_modifier` weights first before committing to updating
    any other weights (in a large fashion).
    """
    MT5_FP16_WARMUP_NUM_STEPS = 100

    # This is an exact copy of `TibToEngTranslationMixin.get_translation_training_arguments` unless specified otherwise
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

            ###########################
            # START mt5 fp16 modification
            ###########################

            # Original code:
            # lr_scheduler_type=SchedulerType.CONSTANT,

            lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            warmup_steps=self.MT5_FP16_WARMUP_NUM_STEPS,

            ###########################
            # END mt5 fp16 modification
            ###########################

            do_train=True,
            do_eval=True,
            seed=42,
            fp16=True,
            log_level="error",
            log_on_each_node=False,
            logging_steps=1,
            predict_with_generate=True,
            deepspeed=self.load_deepspeed_template_args(),
        )



class FinetuneMT5Base580MExperiment(MT5Base580MModelMixin, MT5TibToEngTranslationMixin, FinetuneExperimentBase):
    pass


class FinetuneMT5Large1_2BExperiment(MT5Large1_2BModelMixin, MT5TibToEngTranslationMixin, FinetuneExperimentBase):
    pass
