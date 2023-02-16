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

from torch import rand, randint

from datasets import DatasetDict

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_pipelines import FinetuneExperimentBase

from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.training_argument_mixins.utils import calculate_total_examples

from lme.training_dataset_utils.utils import repeat_examples

from lme.model_mixins import MT5600MModelMixin

from lme.experiments.translation.mixin import TranslationMixin


def truncate_uniformly_randomly(inputs: Dict[str, Sequence], max_input_length: int) -> None:
    # Truncate the labels and append it to the input
    truncation_amount = randint(len(inputs["labels"]), ())
    truncation_amount = min(truncation_amount, max_input_length - len(inputs["input_ids"]))

    to_append = inputs["labels"][:truncation_amount]
    inputs["input_ids"] = inputs["input_ids"] + to_append
    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)


class TranslationIncomplete1Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence]) -> Dict[str, Sequence]:
                if rand(()) < 0.5:
                    pass
                else:
                    truncate_uniformly_randomly(inputs, max_input_length)

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
                truncate_uniformly_randomly(inputs, max_input_length)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete")

        return dataset_dict


class TranslationIncomplete3Mixin(TranslationMixin):
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
                    p_full = 0.2
                elif progress <= 0.6:
                    p_full = progress
                else:
                    p_full = 1

                if rand(()) < p_full:
                    pass
                else:
                    truncate_uniformly_randomly(inputs, max_input_length)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete4Mixin(TranslationMixin):
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
                    truncate_uniformly_randomly(inputs, max_input_length)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete5Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence], idx: int) -> Dict[str, Sequence]:
                progress = idx / total_examples
                max_label_append = max_input_length - len(inputs["input_ids"])
                truncation_amount = int((1 - progress) * max_label_append)

                to_append = inputs["labels"][:truncation_amount]
                inputs["input_ids"] = inputs["input_ids"] + to_append
                inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete6Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)
        TARGET_WARMUP_STEPS = 2000

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence], idx: int) -> Dict[str, Sequence]:
                progress = min(idx / TARGET_WARMUP_STEPS, 1)
                max_label_append = max_input_length - len(inputs["input_ids"])
                truncation_amount = int((1 - progress) * max_label_append)

                to_append = inputs["labels"][:truncation_amount]
                inputs["input_ids"] = inputs["input_ids"] + to_append
                inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncompleteExperimentBase(MT5600MModelMixin, MT5FinetuneArgsMixin, FinetuneExperimentBase):
    pass


class TranslationIncomplete1Experiment(TranslationIncomplete1Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete2Experiment(TranslationIncomplete2Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete3Experiment(TranslationIncomplete3Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete4Experiment(TranslationIncomplete4Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete5Experiment(TranslationIncomplete5Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete6Experiment(TranslationIncomplete6Mixin, TranslationIncompleteExperimentBase):
    pass
