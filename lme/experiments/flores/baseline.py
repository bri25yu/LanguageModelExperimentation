from typing import Callable, Union

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics
from lme.data_processors.flores200 import (
    BaselineSmallDataProcessor, BaselineMediumDataProcessor
)

from lme.model_mixins import MT5300MModelMixin
from lme.training_pipelines import FinetuneExperimentBase
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.training_argument_mixins.utils import calculate_batch_size_args


class FloresBaselineExperimentBase(MT5FinetuneArgsMixin, MT5300MModelMixin, FinetuneExperimentBase):
    MAX_INPUT_LENGTH = 128  # Covers 98.6% of the input dataset
    TRAINER_CLS = Seq2SeqTrainer
    DATA_PROCESSOR_CLS: Union[None, type] = None

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.metric_for_best_model = "chrf++"

        return args

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        data_processor_cls = self.DATA_PROCESSOR_CLS
        return data_processor_cls()(training_arguments)


class FloresBaselineSmallExperiment(FloresBaselineExperimentBase):
    DATA_PROCESSOR_CLS = BaselineSmallDataProcessor


class FloresBaselineMediumExperiment(FloresBaselineExperimentBase):
    DATA_PROCESSOR_CLS = BaselineMediumDataProcessor

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 25000

        target_total_batch_size_per_update = 2 ** 10  # 1024
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        return args


# This is `FloresBaselineMediumExperiment` with the original number of steps i.e. 10k but with a larger batch size
class FloresBaselineMedium2Experiment(FloresBaselineMediumExperiment):
    DATA_PROCESSOR_CLS = BaselineMediumDataProcessor

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 10000

        target_total_batch_size_per_update = 2 ** 11  # 2048
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        return args
