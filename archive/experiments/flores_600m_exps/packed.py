from typing import Callable

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import Packed2DataProcessor, Packed2MixDataProcessor

from lme.model_mixins import MT5600MModelMixin

from lme.experiments.flores_600m_exps.baseline import FloresStagedExperimentBase


class FloresPacked600MExperiment(MT5600MModelMixin, FloresStagedExperimentBase):
    # (2048 / 2) = 1024 // (2 ** 11 / 2 ** 1) = 2 ** 10
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10

    DATA_PROCESSOR_CLASSES = [Packed2DataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass


class FloresPackedMix600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [Packed2MixDataProcessor]
