from typing import Callable

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Seq2SeqTrainer,
)

from lme.data_processors import FinetuneDataProcessor
from lme.training_pipelines.experiment_base import ExperimentBase


__all__ = ["TibToEngTranslationMixin", "TibToEngTranslationWithPrefixMixin"]


class TibToEngTranslationMixin(ExperimentBase):
    """
    Mixin for Tibetan to English translation finetuning
    """
    FINETUNE_TRAINER_CLS = TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_data_collator(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        return self.get_translation_dataset(tokenizer, training_arguments)

    def get_finetune_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_data_collator(tokenizer)

    def get_finetune_dataset(self, tokenizer: PreTrainedTokenizer, finetune_training_arguments: TrainingArguments) -> DatasetDict:
        return self.get_translation_dataset(tokenizer, finetune_training_arguments)

    def get_translation_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset = FinetuneDataProcessor()(training_arguments)

        def tokenize_fn(examples):
            model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        with training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])

        return tokenized_dataset

    def get_translation_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")


class TibToEngTranslationWithPrefixMixin(TibToEngTranslationMixin):
    # This is an exact copy of `TibToEngTranslationMixin.get_translation_dataset` unless specified otherwise
    def get_translation_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset = FinetuneDataProcessor()(training_arguments)

        def tokenize_fn(examples):
            ###########################
            # START add task prefix
            ###########################

            # Original code:
            # model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            prefix = "translate Tibetan to English: "
            tibetan_inputs = [prefix + t for t in examples["tibetan"]]
            model_inputs = tokenizer(tibetan_inputs, max_length=max_input_length, truncation=True)

            ###########################
            # END add task prefix
            ###########################

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs

        with training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])

        return tokenized_dataset
