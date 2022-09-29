from transformers import TrainingArguments

from attention_driven.experiments.finetune_mt5 import FinetuneMT5ExperimentBase


__all__ = ["ZeroShotMT5BaseExperiment", "ZeroShotMT5LargeExperiment", "ZeroShotMT5XLExperiment"]


class ZeroShotMT5ExperimentBase(FinetuneMT5ExperimentBase):
    def get_training_arguments(self, learning_rate: float, batch_size: int) -> TrainingArguments:
        training_arguments = super().get_training_arguments(learning_rate, batch_size)

        training_arguments.max_steps = 0

        return training_arguments


class ZeroShotMT5BaseExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class ZeroShotMT5LargeExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class ZeroShotMT5XLExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
