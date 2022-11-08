from datasets import DatasetDict

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors import TranslationDataProcessor

from lme.training_dataset_utils.tib_to_eng_translation import tokenize_tib_to_eng_translation_with_prefix
from lme.experiments.translation.mixin import TranslationMixin


class TranslationWithPrefixMixin(TranslationMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = TranslationDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_tib_to_eng_translation_with_prefix(translation_dataset, max_input_length, tokenizer, "Translate Tibetan to English")

        return tokenized_dataset
