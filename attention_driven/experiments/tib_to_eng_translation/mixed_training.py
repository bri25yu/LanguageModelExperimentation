from typing import Callable

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments

from attention_driven.data_processors.utils import dataset_summary
from attention_driven.data_processors.pretrain import PretrainDataProcessor

from attention_driven.modeling.t5_span_mlm import (
    compute_input_and_target_lengths, get_group_texts_fn
)

from attention_driven.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationWithPrefixMixin
from attention_driven.experiments.tib_to_eng_translation.tib_zh_eng_pretrain_mixin import TibZhEngPretrainExperimentMixin


class TibToEngWithTibMixin(TibToEngTranslationWithPrefixMixin, TibZhEngPretrainExperimentMixin):
    # This is an exact copy of `TibZhEngPretrainExperimentMixin.get_pretrain_dataset` unless specified otherwise
    def get_pretrain_dataset(self, tokenizer: PreTrainedTokenizer, pretrain_training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH
        mlm_probability = self.MLM_PROBABILITY
        mean_noise_span_length = self.MEAN_NOISE_SPAN_LENGTH

        # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
        # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
        # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
        expanded_inputs_length, _ = compute_input_and_target_lengths(
            inputs_length=max_input_length,
            noise_density=mlm_probability,
            mean_noise_span_length=mean_noise_span_length,
        )
        group_texts = get_group_texts_fn(expanded_inputs_length)

        dataset_dict = PretrainDataProcessor()(pretrain_training_arguments)

        ###############################
        # START only take tibetan
        ###############################

        del dataset_dict["english"]
        del dataset_dict["chinese"]

        ###############################
        # END only take tibetan
        ###############################

        def tokenize_fn(examples):
            tokenized = tokenizer(examples["text"])
            return {"input_ids": tokenized["input_ids"]}

        with pretrain_training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_grouped_dataset_dict = dataset_dict \
                .map(tokenize_fn, batched=True, remove_columns=["text"]) \
                .map(group_texts, batched=True)

            tokenized_group_dataset = concatenate_datasets(list(tokenized_grouped_dataset_dict.values()))

            shuffled_tokenized_grouped_dataset = tokenized_group_dataset.shuffle(seed=42)

            pretrain_dataset = DatasetDict({"train": shuffled_tokenized_grouped_dataset})

        return pretrain_dataset

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        translation_dataset = self.get_translation_dataset(tokenizer, training_arguments)
        monolingual_dataset = self.get_pretrain_dataset(tokenizer, training_arguments)
        self.print_on_main_process_only(training_arguments, dataset_summary(translation_dataset))
        self.print_on_main_process_only(training_arguments, dataset_summary(monolingual_dataset))

        translation_data_collator = self.get_translation_data_collator(tokenizer)
        monolingual_data_collator = self.get_pretrain_data_collator(tokenizer)

        with training_arguments.main_process_first():
            translation_collated = translation_dataset.map(translation_data_collator, batched=True)
            monolingual_collated = monolingual_dataset.map(monolingual_data_collator, batched=True)

            # For now, we use a 1:1 mixing
            mixed_train_dataset: Dataset = concatenate_datasets([translation_collated["train"], monolingual_collated["train"]])
            mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)

            dataset = DatasetDict({
                "train": mixed_train_dataset,
                "val": translation_collated["val"],
                "test": translation_collated["test"],
            })

        return dataset

    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return None
