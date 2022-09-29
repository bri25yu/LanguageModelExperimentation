from attention_driven.experiments.finetune_mt5 import FinetuneMT5ExperimentBase


__all__ = ["ZeroShotMT5BaseExperiment", "ZeroShotMT5LargeExperiment", "ZeroShotMT5XLExperiment"]


class ZeroShotMT5ExperimentBase(FinetuneMT5ExperimentBase):
    NUM_TRAIN_EPOCHS = 0


class ZeroShotMT5BaseExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class ZeroShotMT5LargeExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class ZeroShotMT5XLExperiment(ZeroShotMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
