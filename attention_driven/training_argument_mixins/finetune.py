from transformers import TrainingArguments

from attention_driven.training_argument_mixins.utils import get_seq2seq_training_arguments


__all__ = [
    "ConstantLRFinetuneTrainingArgumentsMixin",
    "DecayLRFinetuneTrainingArgumentsMixin",
]


class ConstantLRFinetuneTrainingArgumentsMixin:
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return get_seq2seq_training_arguments(
            base_output_dir=self.experiment_class_output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            target_total_batch_size_per_update=2 ** 8,  # 256
            per_gpu_batch_size=batch_size,
            scheduler_type="WarmupLR",
            warmup_steps=0,
        )


class DecayLRFinetuneTrainingArgumentsMixin:
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return get_seq2seq_training_arguments(
            base_output_dir=self.experiment_class_output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            target_total_batch_size_per_update=2 ** 8,  # 256
            per_gpu_batch_size=batch_size,
            scheduler_type="WarmupDecayLR",
            warmup_steps=1000,  # 10% of the total number of steps
        )
