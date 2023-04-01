from typing import Callable, List, Union

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics

from lme.data_processors.abstract import AbstractDataProcessor
from lme.data_processors.flores200 import BaselineMediumDataProcessor

from lme.model_mixins import MT51BModelMixin
from lme.training_pipelines import FinetuneExperimentBase
from lme.training_argument_mixins import FloresMT5FinetuneArgsMixin


class FloresBaseline1B20milExperiment(MT51BModelMixin, FloresMT5FinetuneArgsMixin, FinetuneExperimentBase):
    MAX_INPUT_LENGTH = 256
    TRAINER_CLS = Seq2SeqTrainer
    DATA_PROCESSOR_CLASSES: Union[None, List[AbstractDataProcessor]] = [BaselineMediumDataProcessor]

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding=True, pad_to_multiple_of=8)

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        pass

    def get_tokenized_datasets(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        return list(map(lambda c: c()(training_arguments), self.DATA_PROCESSOR_CLASSES))


