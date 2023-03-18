from typing import Callable, List, Union

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics

from lme.data_processors.abstract import AbstractDataProcessor
from lme.data_processors.flores200 import (
    BaselineMediumDataProcessor, PackedDataProcessor
)

from lme.model_mixins import MT5600MModelMixin
from lme.training_pipelines import FinetuneStagedTrainingArgsExperimentBase
from lme.training_argument_mixins import FloresMT5FinetuneArgsMixin


class FloresStagedExperimentBase(FloresMT5FinetuneArgsMixin, FinetuneStagedTrainingArgsExperimentBase):
    MAX_INPUT_LENGTH = 1024
    TRAINER_CLS = Seq2SeqTrainer
    DATA_PROCESSOR_CLASSES: Union[None, List[AbstractDataProcessor]] = None

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding=True, pad_to_multiple_of=8)

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics(tokenizer)

    def get_tokenized_datasets(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> List[DatasetDict]:
        return list(map(lambda c: c()(training_arguments), self.DATA_PROCESSOR_CLASSES))


class FloresBaseline600MExperiment(MT5600MModelMixin, FloresStagedExperimentBase):
    DATA_PROCESSOR_CLASSES = [BaselineMediumDataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass


class FloresPacked600MExperiment(MT5600MModelMixin, FloresStagedExperimentBase):
    # (2048 / 8) = 256 // (2 ** 11 / 2 ** 3) = 2 ** 8
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 8

    DATA_PROCESSOR_CLASSES = [PackedDataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass
