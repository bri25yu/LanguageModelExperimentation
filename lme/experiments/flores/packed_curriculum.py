from transformers import TrainingArguments

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


class FloresPackedCurriculum300MExperiment(FloresPackedCurriculumExperimentBase):
    pass
