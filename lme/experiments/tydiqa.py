from typing import Callable, Dict, Sequence

from torch import randint

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors import TyDiQADataProcessor
from lme.training_dataset_utils.tydiqa import tokenize_tydiqa
from lme.training_dataset_utils.utils import repeat_examples

from lme.model_mixins import (
    MT5600MModelMixin,
    MT51BModelMixin,
    MT53BModelMixin,
)
from lme.training_pipelines import FinetuneExperimentBase
from lme.compute_metrics_utils import get_exact_match_compute_metrics

from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.training_argument_mixins.utils import calculate_total_examples


class TyDiQAExperimentBase(MT5FinetuneArgsMixin, FinetuneExperimentBase):
    MAX_INPUT_LENGTH = 2 ** 9  # Covers 96% of the TyDiQA gold dataset
    TRAINER_CLS = Seq2SeqTrainer

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        # !TODO: This is not proper coding/engineering patterning, but I'm too lazy to invest time to properly fix it.
        training_args = super().get_training_arguments(batch_size, learning_rate)

        training_args.metric_for_best_model = "exact_match"

        return training_args

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_exact_match_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset = TyDiQADataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_tydiqa(dataset, max_input_length, tokenizer)

        return tokenized_dataset


class TyDiQAIncompleteExperimentBase(TyDiQAExperimentBase):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence], idx: int) -> Dict[str, Sequence]:
                progress = idx / total_examples
                if progress <= 0.2:
                    # Truncate the labels and append it to the input
                    truncation_amount = randint(len(inputs["labels"]), ())
                    truncation_amount = min(truncation_amount, max_input_length - len(inputs["input_ids"]))

                    to_append = inputs["labels"][:truncation_amount]
                    inputs["input_ids"] = inputs["input_ids"] + to_append
                    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TyDiQAMT5600MExperiment(MT5600MModelMixin, TyDiQAExperimentBase):
    pass


class TyDiQAMT51BExperiment(MT51BModelMixin, TyDiQAExperimentBase):
    pass


class TyDiQAMT53BExperiment(MT53BModelMixin, TyDiQAExperimentBase):
    pass


class TyDiQAIncompleteMT5600MExperiment(MT5600MModelMixin, TyDiQAIncompleteExperimentBase):
    pass


class TyDiQAIncompleteMT51BExperiment(MT51BModelMixin, TyDiQAIncompleteExperimentBase):
    pass


class TyDiQAIncompleteMT53BExperiment(MT53BModelMixin, TyDiQAIncompleteExperimentBase):
    pass
