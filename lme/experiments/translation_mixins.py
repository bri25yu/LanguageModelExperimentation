from typing import Callable, Union

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils import get_translation_compute_metrics
from lme.data_processors import TranslationDataProcessor, MonolingualDataProcessor
from lme.training_argument_mixins.utils import calculate_total_examples
from lme.training_dataset_utils import (
    create_tib_to_eng_translation,
    create_mix_by_proportion,
    create_examples_proportional_monolingual,
)


__all__ = [
    "TranslationMixin",
    "MixedExamplesProportionalMixin",
    "MixedProportion1Mixin",
    "MixedProportion2Mixin",
    "MixedProportion3Mixin",
]


class TranslationMixin:
    MAX_INPUT_LENGTH = 256  # Covers 96% of the translation dataset
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_translation_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = TranslationDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = create_tib_to_eng_translation(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset


class MixedMixinBase(TranslationMixin):
    TRANSLATION_PROPORTION: Union[None, float] = None
    MONOLINGUAL_PROPORTION: Union[None, float] = None

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH
        translation_proportion = self.TRANSLATION_PROPORTION
        monolingual_proportion = self.MONOLINGUAL_PROPORTION

        total_examples = calculate_total_examples(training_arguments)

        translation_dataset = TranslationDataProcessor()(training_arguments)
        monolingual_dataset = MonolingualDataProcessor()(training_arguments)
        monolingual_dataset = DatasetDict({"tibetan": monolingual_dataset["tibetan"]})

        with training_arguments.main_process_first():
            tokenized_translation_dataset = create_tib_to_eng_translation(translation_dataset, max_input_length, tokenizer)
            monolingual_dataset = create_examples_proportional_monolingual(tokenizer, max_input_length, monolingual_dataset)

            mixed_train_dataset = create_mix_by_proportion(
                tokenized_translation_dataset["train"],
                monolingual_dataset["train"],
                tokenizer,
                max_input_length,
                total_examples,
                translation_proportion,
                monolingual_proportion,
            )

            mixed_dataset = DatasetDict({
                "train": mixed_train_dataset,
                "val": tokenized_translation_dataset["val"],
                "test": tokenized_translation_dataset["test"],
            })

        return mixed_dataset


class MixedProportion1Mixin(MixedMixinBase):
    TRANSLATION_PROPORTION = 0.1
    MONOLINGUAL_PROPORTION = 0.9


class MixedProportion2Mixin(MixedMixinBase):
    TRANSLATION_PROPORTION = 0.25
    MONOLINGUAL_PROPORTION = 0.75


class MixedProportion3Mixin(MixedMixinBase):
    TRANSLATION_PROPORTION = 0.5
    MONOLINGUAL_PROPORTION = 0.5
