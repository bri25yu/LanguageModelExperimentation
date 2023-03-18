from typing import Callable

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import BaselineMediumDataProcessor, PretrainMixDataProcessor

from lme.experiments.flores.packed_curriculum import FloresPackedCurriculumExperimentBase


class FloresPretrainMixExperimentBase(FloresPackedCurriculumExperimentBase):
    MAX_INPUT_LENGTH = 128
    DATA_PROCESSOR_CLASSES = [
        PretrainMixDataProcessor,
        BaselineMediumDataProcessor,
    ]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        if stage == 1:
            max_steps = 4000
        elif stage == 2:
            max_steps = 10000
        else:
            raise ValueError(f"Unknown stage {stage}")

        training_arguments.max_steps = max_steps

        training_arguments.__post_init__()  # Recreate hf deepspeed config

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass


class TestFloresPretrainMixExperiment(FloresPretrainMixExperimentBase):
    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        super().update_training_arguments(training_arguments, batch_size, stage)

        if stage == 1:
            max_steps = 400
        elif stage == 2:
            max_steps = 800

        training_arguments.max_steps = max_steps


class FloresPretrainMix300MExperiment(FloresPretrainMixExperimentBase):
    pass
