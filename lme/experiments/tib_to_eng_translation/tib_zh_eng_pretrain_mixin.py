from typing import Callable

import os

from datasets import DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    TrainingArguments, Seq2SeqTrainer
)

from lme.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin, TibToEngTranslationWithPrefixMixin
from lme.data_processors import PretrainDataProcessor
from lme.modeling.t5_span_mlm import (
    PyTorchDataCollatorForT5MLM, compute_input_and_target_lengths, get_group_texts_fn
)


__all__ = ["TibZhEngPretrainExperimentMixin", "TibZhEngPretrainWithPrefixExperimentMixin"]


class TibZhEngPretrainExperimentMixin(TibToEngTranslationMixin):
    """
    This is specific to mT5, specifically we use a T5-style span masking
    data collator in `get_pretrain_data_collator` and do T5-style span masking
    preprocessing in `get_pretrain_dataset`.
    """
    PRETRAIN_TRAINER_CLS = Seq2SeqTrainer

    # T5-style span masking parameters
    MLM_PROBABILITY = 0.15
    MEAN_NOISE_SPAN_LENGTH = 3.0

    def get_pretrain_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH
        mlm_probability = self.MLM_PROBABILITY
        mean_noise_span_length = self.MEAN_NOISE_SPAN_LENGTH

        _, targets_length = compute_input_and_target_lengths(
            inputs_length=max_input_length,
            noise_density=mlm_probability,
            mean_noise_span_length=mean_noise_span_length,
        )

        return PyTorchDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=mlm_probability,
            mean_noise_span_length=mean_noise_span_length,
            input_length=max_input_length,
            target_length=targets_length,
        )

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


class TibZhEngPretrainWithPrefixExperimentMixin(TibZhEngPretrainExperimentMixin):
    """
    There's no change between the pretrain strategies, only with the finetune dataset creation.
    """
    get_translation_dataset = TibToEngTranslationWithPrefixMixin.get_translation_dataset
