"""
Incomplete1
    The baseline experiment with no changes to the input.

Incomplete2
    50% chance for the sequence to be unaltered. The other 50% of the time, we truncate the
    labels and append it to the input. The amount of truncation here is uniformly random.
    For sequences that are longer than the max_length, we append as many tokens as possible.

Incomplete3
    Truncate the labels and append it to the input. The amount of truncation here is uniformly random.
    For sequences that are longer than the max_length, we append as many tokens as possible.

"""
from typing import Dict, Sequence

from torch import randint

from datasets import DatasetDict

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_pipelines import FinetuneExperimentBase

from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.training_argument_mixins.utils import calculate_total_examples

from lme.training_dataset_utils.utils import repeat_examples

from lme.model_mixins import MT5Base580MModelMixin

from lme.experiments.translation.mixin import TranslationMixin


class TranslationIncomplete1Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence]) -> Dict[str, Sequence]:
                to_truncate = randint(2, ())

                if not to_truncate:
                    # 50% of the time, we leave the sequence as is
                    pass
                else:
                    # The other 50% of the time, we truncate the labels and append it to the input
                    # The amount of truncation here is uniformly random
                    truncation_amount = randint(len(inputs["labels"]), ())
                    truncation_amount = min(truncation_amount, max_input_length - len(inputs["input_ids"]))

                    to_append = inputs["labels"][:truncation_amount]
                    inputs["input_ids"] = inputs["input_ids"] + to_append
                    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete")

        return dataset_dict


class TranslationIncomplete2Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence]) -> Dict[str, Sequence]:
                # Truncate the labels and append it to the input
                # The amount of truncation here is uniformly random
                truncation_amount = randint(len(inputs["labels"]), ())
                truncation_amount = min(truncation_amount, max_input_length - len(inputs["input_ids"]))

                to_append = inputs["labels"][:truncation_amount]
                inputs["input_ids"] = inputs["input_ids"] + to_append
                inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete")

        return dataset_dict


class TranslationIncompleteExperimentBase(MT5Base580MModelMixin, MT5FinetuneArgsMixin, FinetuneExperimentBase):
    pass


class TranslationIncomplete1Experiment(TranslationIncomplete1Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete2Experiment(TranslationIncomplete2Mixin, TranslationIncompleteExperimentBase):
    pass
