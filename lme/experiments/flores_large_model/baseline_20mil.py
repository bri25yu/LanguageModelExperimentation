from typing import Callable, List, Union

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics

from lme.data_processors.abstract import AbstractDataProcessor
from lme.data_processors.flores200 import BaselineMediumDataProcessor
from lme.experiments.flores_600m_exps.baseline import FloresStagedExperimentBase

from lme.model_mixins import MT51BModelMixin
from lme.training_pipelines import FinetuneExperimentBase
from lme.training_argument_mixins import FloresMT5FinetuneArgsMixin


class FloresBaseline1BExperiment(MT51BModelMixin, FloresStagedExperimentBase):
    DATA_PROCESSOR_CLASSES = [BaselineMediumDataProcessor]

    def update_training_arguments(self, training_arguments: TrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass

