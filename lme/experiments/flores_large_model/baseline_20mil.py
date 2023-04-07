from typing import Callable

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import BaselineMediumDataProcessor
from lme.model_mixins import MT51BModelMixin

from lme.experiments.flores_600m_exps.baseline import FloresStagedExperimentBase


class FloresBaseline1BExperiment(MT51BModelMixin, FloresStagedExperimentBase):
    DATA_PROCESSOR_CLASSES = [BaselineMediumDataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass
