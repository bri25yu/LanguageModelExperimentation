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