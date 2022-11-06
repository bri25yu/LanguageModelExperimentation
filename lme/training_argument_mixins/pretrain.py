from typing import Union

from transformers import TrainingArguments

from lme.training_argument_mixins.utils import get_seq2seq_training_arguments
from lme.training_argument_mixins.finetune import (
    ConstantLRFinetuneTrainingArgumentsMixin,
)


__all__ = [
    "PretrainTrainingArgumentsMixin1",
]


class PretrainTrainingArgumentsMixinBase:
    FINETUNE_TRAINING_ARGUMENTS_MIXIN: Union[None, type] = None

    def get_finetune_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return self.FINETUNE_TRAINING_ARGUMENTS_MIXIN.get_training_arguments(
            self, batch_size, learning_rate
        )


class PretrainTrainingArgumentsMixin1(PretrainTrainingArgumentsMixinBase):
    """
    This results in 100k * 256 = 25 mil examples.
    """
    FINETUNE_TRAINING_ARGUMENTS_MIXIN = ConstantLRFinetuneTrainingArgumentsMixin

    def get_pretrain_training_arguments(self, batch_size: int) -> TrainingArguments:
        return get_seq2seq_training_arguments(
            base_output_dir=self.experiment_class_output_dir,
            learning_rate=1e-4,
            max_steps=100000,
            eval_steps=1000,
            target_total_batch_size_per_update=2 ** 8,  # 256
            per_gpu_batch_size=batch_size,
            scheduler_type="WarmupDecayLR",
            warmup_steps=10000,  # 10% of the total number of steps
        )
