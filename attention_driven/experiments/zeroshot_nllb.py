from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


__all__ = ["ZeroShotNLLB600MExperiment", "ZeroShotNLLB1_3BExperiment", "ZeroShotNLLB3_3BExperiment"]


class ZeroShotNLLBExperimentBase(BaselineV2Experiment):
    NUM_TRAIN_EPOCHS = 0


class ZeroShotNLLB600MExperiment(ZeroShotNLLBExperimentBase):
    pass


class ZeroShotNLLB1_3BExperiment(ZeroShotNLLBExperimentBase):
    MODEL_NAME = "facebook/nllb-200-1.3B"


class ZeroShotNLLB3_3BExperiment(ZeroShotNLLBExperimentBase):
    MODEL_NAME = "facebook/nllb-200-3.3B"
