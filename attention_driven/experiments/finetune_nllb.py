from attention_driven.experiments.baseline_v2 import BaselineV2Experiment


__all__ = ["FinetuneNLLB1_3BExperiment", "FinetuneNLLB3_3BExperiment"]


class FinetuneNLLB1_3BExperiment(BaselineV2Experiment):
    MODEL_NAME = "facebook/nllb-200-1.3B"


class FinetuneNLLB3_3BExperiment(BaselineV2Experiment):
    MODEL_NAME = "facebook/nllb-200-3.3B"
