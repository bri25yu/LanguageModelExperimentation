"""
Incomplete1 - 23.9 BLEU
    50% no addition, 50% uniformly distributed addition.
    Less variance in the first 2000 steps, most consistently better than baseline

Incomplete2 - 21.1 BLEU
    Uniformly distributed addition.
    Less variance in the first 2000 steps, dataset later on is not challenging enough to peak properly
    Less steep learning in the inital 2000 steps

Incomplete3 - 24.7 BLEU
    20% no addition/80% uniform, 20-60% linear no addition, 100% no addition/0% uniform
    A little shakier in the first 2000 steps compared to incomplete 1/2
    More variance in the learning curves ://

Incomplete 4 - 24.6 BLEU
    First 2000 steps uniformly distributed addition, rest no addition
    Little struggle in the beginning to learn quickly, but still faster than baseline
    Super consistent among LRs

Incomplete 5 - We don't talk about this one 17.4 BLEU
    Linear addition from 100% to 0%
    Yeah this one is definitely not hard enough

Incomplete 6 - 24.9 BLEU
    Linear addition from 100% to 0% for the first 2000 steps, rest no addition
    Not sure why, but for some reason LR 1e-3 works but 2e-3 fails to train i.e. the loss spikes to inf and the BLEU score drops to 0. Not sure why
    Very varied initial performance unfortunately.

Incomplete 7 - 
    First 2000 steps uniformly distributed addition of suffix and prefix, rest no addition
    Peforms similarly to the other best incomplete experiments.
    For one ablation ran into some unknown issue with a very large spike in loss and BLEU score dropping, then it recovers. 

Incomplete 8 - 
    First 4000 steps uniformly distributed addition of suffix and prefix (same as #7)


Incomplete 9 - 
    First 1200 steps uniformly distributed addition of suffix and prefix (same as #7), rest no addition
    Potentially making the task easier for a lesser amount of the training is helpful for the model to then adapt to the task more quickly
    during the rest of training. 

Incomplete 10 - 
    First 2000 steps uniformly distributed addition of masked target added to the input

Incomplete 11 - 
    First 2000 steps uniformly distributed addition, rest no addition, same as #4. 
    Last 2000 steps, mask portions of the input (not the target), so removing information from the input. Perhaps this more difficult task will
    help the model to learn better.
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
from lme.training_dataset_utils.incomplete_utils import add_prefix_truncated_output, add_prefix_and_suffix_truncated_output, add_middle_truncated_output, add_suffix_truncated_output, add_masked_output

from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin

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
                if rand(()) < 0.5:
                    pass
                else:
                    add_prefix_truncated_output(inputs, max_input_length)

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
                add_prefix_truncated_output(inputs, max_input_length)

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
                    add_prefix_truncated_output(inputs, max_input_length)

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
                    add_prefix_truncated_output(inputs, max_input_length)

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


class TranslationIncomplete7Mixin(TranslationMixin):
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
                    add_prefix_and_suffix_truncated_output(inputs, max_input_length)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete8Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence], idx: int) -> Dict[str, Sequence]:
                progress = idx / total_examples
                if progress <= 0.4:
                    add_prefix_and_suffix_truncated_output(inputs, max_input_length, tokenizer)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete9Mixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        dataset_dict = super().get_tokenized_dataset(tokenizer, training_arguments)

        max_input_length = self.MAX_INPUT_LENGTH
        total_examples = calculate_total_examples(training_arguments)

        with training_arguments.main_process_first():
            train_dataset = dataset_dict["train"]
            train_dataset = repeat_examples(train_dataset, total_examples)

            def map_fn(inputs: Dict[str, Sequence], idx: int) -> Dict[str, Sequence]:
                progress = idx / total_examples
                if progress <= 0.12:
                    add_prefix_and_suffix_truncated_output(inputs, max_input_length)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete10Mixin(TranslationMixin):
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
                    add_masked_output(inputs, max_input_length, tokenizer)

                return inputs

            dataset_dict["train"] = train_dataset.map(map_fn, desc="Applying incomplete", with_indices=True)

        return dataset_dict


class TranslationIncomplete11Mixin(TranslationMixin):
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
                    # TODO 
                    add_masked_output(inputs, max_input_length, tokenizer)

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


class TranslationIncomplete7Experiment(TranslationIncomplete7Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete8Experiment(TranslationIncomplete8Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete9Experiment(TranslationIncomplete9Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete10Experiment(TranslationIncomplete10Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncomplete11Experiment(TranslationIncomplete11Mixin, TranslationIncompleteExperimentBase):
    pass


class TranslationIncompleteMT51BExperiment(
    TranslationIncomplete4Mixin, MT51BModelMixin, MT5FinetuneArgsMixin, FinetuneExperimentBase
):
    pass


class TranslationIncompleteMT53BExperiment(
    TranslationIncomplete4Mixin, MT53BModelMixin, MT5FinetuneArgsMixin, FinetuneExperimentBase
):
    pass
