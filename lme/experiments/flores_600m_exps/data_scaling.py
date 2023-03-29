from typing import Callable

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import BaselineAllDataProcessor

from lme.model_mixins import MT5600MModelMixin

from lme.experiments.flores_600m_exps.baseline import FloresStagedExperimentBase


class FloresDataScaling600MExperimentBase(MT5600MModelMixin, FloresStagedExperimentBase):
    # Baseline is 20M examples seen
    DATA_PROCESSOR_CLASSES = [BaselineAllDataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass


class FloresDataScaling600M40MExperiment(FloresDataScaling600MExperimentBase):
    # Baseline is 2048 batch size. 
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 12  # 2048 * 2 = 4096; 2 ** 11 * 2 ** 1 = 2 ** 12


class FloresDataScaling600M80MExperiment(FloresDataScaling600MExperimentBase):
    # Baseline is 2048 batch size. 
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 13  # 2048 * 4 = 8192; 2 ** 11 * 2 ** 2 = 2 ** 13
