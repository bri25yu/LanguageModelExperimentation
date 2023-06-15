from transformers import TrainingArguments

from lme.data_processors.flores200 import PackedDataProcessor

from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.experiments.flores_300m_exps.baseline import FloresBaselineMedium2Experiment


class FloresPackedExperimentBase(FloresBaselineMedium2Experiment):
    DATA_PROCESSOR_CLS = PackedDataProcessor
    MAX_INPUT_LENGTH = 1024

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 10000

        target_total_batch_size_per_update = 2 ** 8  # (2048 / 8) = 256 or (2 ** 11 / 2 ** 3) = 2 ** 8
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        args.__post_init__()  # Reload hf deepspeed config

        return args


class FloresPacked300MExperiment(FloresPackedExperimentBase):
    pass
