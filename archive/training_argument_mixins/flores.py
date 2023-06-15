import os

from transformers import TrainingArguments, Seq2SeqTrainingArguments

from lme.training_argument_mixins.utils import (
    get_deepspeed_args, calculate_batch_size_args, get_default_training_arguments
)


__all__ = [
    "FloresMT5FinetuneLargeArgsMixin",
]


class FloresMT5FinetuneLargeArgsMixin:
    """
    This results in 20k * 2048 = 40mil examples.
    """
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 11  # 2048

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )
        target_total_batch_size_per_update = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE

        # !TODO: This is very poor engineering and programming practice. Sue me
        use_bf16 = os.environ.get("USE_BF16", "true")
        use_bf16 = use_bf16 == "true"

        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=20000,
            eval_steps=200,
            save_steps=200,
            warmup_steps=0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=False,
            bf16=use_bf16,
            metric_for_best_model="chrf++",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )
