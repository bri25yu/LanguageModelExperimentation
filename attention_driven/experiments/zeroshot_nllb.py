from transformers import TrainingArguments

from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


__all__ = ["ZeroShotNLLB600MExperiment", "ZeroShotNLLB1_3BExperiment", "ZeroShotNLLB3_3BExperiment"]


class ZeroShotNLLBExperimentBase(BaselineV2Experiment):
    def get_training_arguments(self, learning_rate: float, batch_size: int) -> TrainingArguments:
        training_arguments = super().get_training_arguments(learning_rate, batch_size)

        training_arguments.do_train = False
        training_arguments.deepspeed = None

        return training_arguments


class ZeroShotNLLB600MExperiment(ZeroShotNLLBExperimentBase):
    pass


class ZeroShotNLLB1_3BExperiment(ZeroShotNLLBExperimentBase):
    MODEL_NAME = "facebook/nllb-200-1.3B"


class ZeroShotNLLB3_3BExperiment(ZeroShotNLLBExperimentBase):
    MODEL_NAME = "facebook/nllb-200-3.3B"
