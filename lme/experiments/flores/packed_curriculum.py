from typing import Callable

from datasets import Dataset

from transformers import DataCollatorForSeq2Seq, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors.flores200 import PackedCurriculumDataProcessor
from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.training_pipelines import FinetuneStagedTrainingArgsExperimentBase

from lme.experiments.flores.packed import FloresPackedExperimentBase


class FloresPackedCurriculumExperimentBase(FinetuneStagedTrainingArgsExperimentBase, FloresPackedExperimentBase):
    DATA_PROCESSOR_CLS = PackedCurriculumDataProcessor

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 2000

        target_total_batch_size_per_update = 2 ** 11  # 2048, the baseline batch size
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        return args

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int) -> TrainingArguments:
        training_arguments.max_steps = 10000

        target_total_batch_size_per_update = 2 ** 8  # 2048 / 8 = 256
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        training_arguments.gradient_accumulation_steps = gradient_accumulation_steps
        training_arguments.per_device_train_batch_size = per_device_batch_size
        training_arguments.per_device_eval_batch_size = 2 * per_device_batch_size

        return training_arguments

    def create_stage2_training_dataset(self, training_dataset: Dataset) -> Dataset:
        # Skip past the datapoints already trained on that the trainer won't skip
        # (num_old_steps * old_batch_size) - (num_old_steps * new_batch_size)
        num_old_steps = 2000  # From `get_training_arguments` `max_steps`
        old_batch_size = 2 ** 11
        new_batch_size = 2 ** 8

        total_to_skip = num_old_steps * old_batch_size
        trainer_skipped = num_old_steps * new_batch_size
        manually_skip = total_to_skip - trainer_skipped

        total_n_points = len(training_dataset)
        return training_dataset.select(range(manually_skip, total_n_points))

    # First stage data collator uses sequence length of 128
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = 128

        # We use `padding=True` here to save on some compute
        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding=True)

    # Second stage data collator uses sequence length of 1024
    def get_stage2_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = 1024

        # We use `padding=True` here to save on some compute
        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding=True)


class FloresPackedCurriculum300MExperiment(FloresPackedCurriculumExperimentBase):
    pass
