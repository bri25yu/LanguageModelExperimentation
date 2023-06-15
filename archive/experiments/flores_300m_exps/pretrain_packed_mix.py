from typing import Callable

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.flores200 import PackedDataProcessor, PretrainPackedMixDataProcessor

from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.experiments.flores_300m_exps.packed_curriculum import FloresPackedCurriculumExperimentBase


class FloresPretrainPackedMixExperimentBase(FloresPackedCurriculumExperimentBase):
    MAX_INPUT_LENGTH = 1024
    DATA_PROCESSOR_CLASSES = [
        PretrainPackedMixDataProcessor,
        PackedDataProcessor,
    ]

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 10000

        target_total_batch_size_per_update = 2 ** 8  # 2048
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        args.__post_init__()

        return args

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        if stage == 1:
            max_steps = 2000
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


class FloresPretrainPackedMix300MExperiment(FloresPretrainPackedMixExperimentBase):
    pass
