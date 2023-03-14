from typing import Callable

from datasets import Dataset

from transformers import TrainingArguments, Seq2SeqTrainer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import PackedCurriculumDataProcessor
from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.training_pipelines import FinetuneStagedTrainingArgsExperimentBase

from lme.experiments.flores.packed import FloresPackedExperimentBase


class NoShufflingSeq2SeqTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self):
        train_sampler = super()._get_train_sampler()

        if train_sampler is not None and hasattr(train_sampler, "shuffle"):
            train_sampler.shuffle = False

        return train_sampler


class FloresPackedCurriculumExperimentBase(FinetuneStagedTrainingArgsExperimentBase, FloresPackedExperimentBase):
    DATA_PROCESSOR_CLS = PackedCurriculumDataProcessor
    TRAINER_CLS = NoShufflingSeq2SeqTrainer

    STAGE1_SEQUENCE_LENGTH = 128
    STAGE2_SEQUENCE_LENGTH = 1024

    STAGE1_TARGET_BATCH_SIZE = 2 ** 11  # 2048
    STAGE2_TARGET_BATCH_SIZE = 2 ** 8   # (2048/8) = 256

    STAGE1_MAX_STEPS = 2000
    STAGE2_MAX_STEPS = 10000

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = self.STAGE1_MAX_STEPS

        stage1_batch_size_factor = 8  # Approximate
        batch_size = batch_size * stage1_batch_size_factor

        target_total_batch_size_per_update = self.STAGE1_TARGET_BATCH_SIZE
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        return args

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int) -> TrainingArguments:
        training_arguments.max_steps = self.STAGE2_MAX_STEPS

        target_total_batch_size_per_update = self.STAGE2_TARGET_BATCH_SIZE
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        training_arguments.gradient_accumulation_steps = gradient_accumulation_steps
        training_arguments.per_device_train_batch_size = per_device_batch_size
        training_arguments.per_device_eval_batch_size = 2 * per_device_batch_size

        return training_arguments

    def create_stage2_training_dataset(self, training_dataset: Dataset) -> Dataset:
        # Skip past the datapoints already trained on that the trainer won't skip
        # (num_old_steps * old_batch_size) - (num_old_steps * new_batch_size)
        num_old_steps = self.STAGE1_MAX_STEPS
        old_batch_size = self.STAGE1_TARGET_BATCH_SIZE
        new_batch_size = self.STAGE2_TARGET_BATCH_SIZE

        total_to_skip = num_old_steps * old_batch_size
        trainer_skipped = num_old_steps * new_batch_size
        manually_skip = total_to_skip - trainer_skipped

        total_n_points = len(training_dataset)
        return training_dataset.select(range(manually_skip, total_n_points))

    # First stage data collator uses sequence length of 128
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        data_collator = super().get_data_collator(tokenizer)

        max_input_length = self.STAGE1_SEQUENCE_LENGTH

        data_collator.max_length = max_input_length

        return data_collator

    # Second stage data collator uses sequence length of 1024
    def get_stage2_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        data_collator = super().get_data_collator(tokenizer)

        max_input_length = self.STAGE2_SEQUENCE_LENGTH

        data_collator.max_length = max_input_length

        return data_collator

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model = super().get_model(tokenizer)

        max_input_length = self.STAGE1_SEQUENCE_LENGTH

        model.config.max_length = max_input_length

        return model

    def update_model(self, model: PreTrainedModel) -> None:
        max_input_length = self.STAGE2_SEQUENCE_LENGTH

        model.config.max_length = max_input_length


class FloresPackedCurriculum300MExperiment(FloresPackedCurriculumExperimentBase):
    pass
