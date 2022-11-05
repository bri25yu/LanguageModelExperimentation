from transformers import TrainingArguments

from lme.training_argument_mixins.utils import get_seq2seq_training_arguments


__all__ = ["PretrainTrainingArgumentsMixin"]


class PretrainTrainingArgumentsMixin:
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return get_seq2seq_training_arguments(
            base_output_dir=self.experiment_class_output_dir,
            learning_rate=learning_rate,
            max_steps=100000,
            eval_steps=1000,
            target_total_batch_size_per_update=2 ** 10,  # 1024
            per_gpu_batch_size=batch_size,
            scheduler_type="WarmupDecayLR",
            warmup_steps=10000,  # 10% of the total number of steps
        )
