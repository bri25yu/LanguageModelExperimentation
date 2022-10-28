from typing import Callable

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments

from attention_driven.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationWithPrefixMixin
from attention_driven.experiments.tib_to_eng_translation.tib_zh_eng_pretrain_mixin import TibZhEngPretrainExperimentMixin


class TibToEngWithTibMixin(TibToEngTranslationWithPrefixMixin, TibZhEngPretrainExperimentMixin):
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        translation_dataset = self.get_translation_dataset(tokenizer, training_arguments)
        monolingual_dataset = self.get_pretrain_dataset(tokenizer, training_arguments)

        translation_data_collator = self.get_translation_data_collator(tokenizer)
        monolingual_data_collator = self.get_pretrain_data_collator(tokenizer)

        tibetan_monolingual = monolingual_dataset["tibetan"]

        with training_arguments.main_process_first():
            translation_collated = translation_dataset.map(translation_data_collator, batched=True)
            tibetan_collated = tibetan_monolingual.map(monolingual_data_collator, batched=True)

            mixed_train_dataset: Dataset = concatenate_datasets([translation_collated["train"], tibetan_collated])
            mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)

            dataset = DatasetDict({
                "train": mixed_train_dataset,
                "val": translation_collated["val"],
                "test": translation_collated["test"],
            })

        return dataset

    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return None
