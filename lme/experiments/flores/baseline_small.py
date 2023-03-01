from typing import Callable

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics
from lme.data_processors.flores200.baseline_small import BaselineSmallDataProcessor

from lme.model_mixins import MT5300MModelMixin
from lme.training_pipelines import FinetuneExperimentBase
from lme.training_argument_mixins import MT5FinetuneArgsMixin


class FloresBaselineSmallExperiment(MT5FinetuneArgsMixin, MT5300MModelMixin, FinetuneExperimentBase):
    MAX_INPUT_LENGTH = 128  # Covers 98.6% of the input dataset
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        return BaselineSmallDataProcessor()(training_arguments)
