from typing import Any, Dict

import os

from transformers import TrainingArguments, Seq2SeqTrainingArguments

from lme.training_argument_mixins.utils import (
    get_deepspeed_args, calculate_batch_size_args, get_default_training_arguments
)


__all__ = [
    "MT5FinetuneArgsMixin",
    "NLLBFinetuneArgsMixin",
    "BloomFinetuneArgsMixin",
]


class MT5FinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            save_steps=200,
            warmup_steps=0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            fp16=False,
            bf16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )


class NLLBFinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            save_steps=200,
            warmup_steps=1000,  # 10% of the total number of steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            fp16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupDecayLR"),
            **get_default_training_arguments(),
        )


class BloomFinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            save_steps=200,
            warmup_steps=1000,  # 10% of the total number of steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            fp16=False,
            bf16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupDecayLR"),
            **get_default_training_arguments(),
        )
