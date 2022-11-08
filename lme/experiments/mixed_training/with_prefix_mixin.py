from typing import Union

from datasets import DatasetDict

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors import TranslationDataProcessor, MonolingualDataProcessor
from lme.training_argument_mixins.utils import calculate_total_examples

from lme.training_dataset_utils.tib_to_eng_translation import tokenize_tib_to_eng_translation_with_prefix
from lme.training_dataset_utils.monolingual import tokenize_tibetan_monolingual_with_prefix
from lme.training_dataset_utils.tib_translation_mix import create_mix_by_proportion

from lme.experiments.translation.mixin import TranslationMixin


class MixedWithPrefixMixinBase(TranslationMixin):
    TRANSLATION_PROPORTION: Union[None, float] = None

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH
        translation_proportion = self.TRANSLATION_PROPORTION

        total_examples = calculate_total_examples(training_arguments)

        translation_dataset = TranslationDataProcessor()(training_arguments)
        monolingual_dataset = MonolingualDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            # Tokenize and add prefixes to our datasets
            translation_dataset = tokenize_tib_to_eng_translation_with_prefix(
                translation_dataset, max_input_length, tokenizer, "Fill in the blank with Tibetan"
            )
            monolingual_train_set = tokenize_tibetan_monolingual_with_prefix(
                monolingual_dataset["tibetan"], max_input_length, tokenizer, "Translate Tibetan to English"
            )

            mixed_train_dataset = create_mix_by_proportion(
                translation_dataset["train"], monolingual_train_set, tokenizer, max_input_length, total_examples, translation_proportion
            )

            mixed_dataset = DatasetDict({
                "train": mixed_train_dataset,
                "val": translation_dataset["val"],
                "test": translation_dataset["test"],
            })

        return mixed_dataset


class MixedWithPrefixProportion1Mixin(MixedWithPrefixMixinBase):
    TRANSLATION_PROPORTION = 0.9


class MixedWithPrefixProportion2Mixin(MixedWithPrefixMixinBase):
    TRANSLATION_PROPORTION = 0.75


class MixedWithPrefixProportion3Mixin(MixedWithPrefixMixinBase):
    TRANSLATION_PROPORTION = 0.5
