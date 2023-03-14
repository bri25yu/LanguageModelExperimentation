from typing import Callable, List

from datasets import DatasetDict

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import BaselineMediumDataProcessor, PackedDataProcessor
from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.training_pipelines import FinetuneStagedTrainingArgsExperimentBase

from lme.experiments.flores.packed import FloresPackedExperimentBase


class FloresPackedCurriculumExperimentBase(FinetuneStagedTrainingArgsExperimentBase, FloresPackedExperimentBase):
    DATA_PROCESSOR_CLASSES = [
        BaselineMediumDataProcessor,
        PackedDataProcessor,
    ]

    def get_tokenized_datasets(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> List[DatasetDict]:
        return list(map(lambda c: c()(training_arguments, self.DATA_PROCESSOR_CLASSES)))

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        if stage == 1:
            max_steps = 2000

            stage1_batch_size_factor = 8  # Approximate
            batch_size = batch_size * stage1_batch_size_factor

            target_total_batch_size_per_update = 2 ** 11  # 2048
        elif stage == 2:
            max_steps = 10000

            target_total_batch_size_per_update = 2 ** 8  # (2048 / 8) = 256
        else:
            raise ValueError(f"Unknown stage {stage}")

        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        training_arguments.max_steps = max_steps
        training_arguments.gradient_accumulation_steps = gradient_accumulation_steps
        training_arguments.per_device_train_batch_size = per_device_batch_size
        training_arguments.per_device_eval_batch_size = 2 * per_device_batch_size

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        if stage == 1:
            max_length = 128
        elif stage == 2:
            max_length = 1024

        data_collator.max_length = max_length

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        if stage == 1:
            max_length = 128
        elif stage == 2:
            max_length = 1024

        model.config.max_length = max_length


class TestFloresPackedCurriculumExperiment(FloresPackedCurriculumExperimentBase):
    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        super().update_training_arguments(training_arguments, batch_size, stage)

        if stage == 1:
            max_steps = 400
        elif stage == 2:
            max_steps = 800

        training_arguments.max_steps = max_steps


class FloresPackedCurriculum300MExperiment(FloresPackedCurriculumExperimentBase):
    pass
