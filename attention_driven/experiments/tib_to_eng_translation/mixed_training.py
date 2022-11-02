from typing import Callable, Tuple

from itertools import cycle

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from attention_driven.data_processors.pretrain import PretrainDataProcessor

from attention_driven.modeling.t5_span_mlm import (
    compute_input_and_target_lengths, get_group_texts_fn
)

from attention_driven.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationWithPrefixMixin
from attention_driven.experiments.tib_to_eng_translation.tib_zh_eng_pretrain_mixin import TibZhEngPretrainExperimentMixin


__all__ = ["TibToEngWithTibMixin", "LongContextMixedTrainingMixin"]


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
            ###############################
            # START return attention mask
            ###############################

            # Original code:
            # tokenized = tokenizer(examples["text"])
            # return {"input_ids": tokenized["input_ids"]}

            return tokenizer(examples["text"])

            ###############################
            # END return attention mask
            ###############################

        with pretrain_training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_grouped_dataset_dict = dataset_dict \
                .map(tokenize_fn, batched=True, remove_columns=["text"]) \
                .map(group_texts, batched=True)

            tokenized_group_dataset = concatenate_datasets(list(tokenized_grouped_dataset_dict.values()))

            shuffled_tokenized_grouped_dataset = tokenized_group_dataset.shuffle(seed=42)

            pretrain_dataset = DatasetDict({"train": shuffled_tokenized_grouped_dataset})

        return pretrain_dataset

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        monolingual_data_collator = self.get_pretrain_data_collator(tokenizer)
        monolingual_data_collator.return_tensors = "np"

        translation_dataset = self.get_translation_dataset(tokenizer, training_arguments)
        monolingual_dataset = self.get_pretrain_dataset(tokenizer, training_arguments)

        with training_arguments.main_process_first():
            translation_dataset, monolingual_dataset = self._create_mix(translation_dataset, monolingual_dataset)

            monolingual_collated = monolingual_dataset.map(monolingual_data_collator, batched=True)

            mixed_train_dataset: Dataset = concatenate_datasets([translation_dataset["train"], monolingual_collated["train"]])
            mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)

            dataset = DatasetDict({
                "train": mixed_train_dataset,
                "val": translation_dataset["val"],
                "test": translation_dataset["test"],
            })

        return dataset

    # Take all the translation data points and fill in the rest with monolingual data points
    # Out of 2,560,000 examples, the model sees 300,000 translation data points
    # and 2,260,000 monolingual examples
    def _create_mix(self, translation_dataset: DatasetDict, monolingual_dataset: DatasetDict) -> Tuple[DatasetDict, DatasetDict]:
        num_train_steps = self.NUM_TRANSLATION_TRAIN_STEPS
        num_examples_per_train_step = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE

        total_examples = num_train_steps * num_examples_per_train_step
        translation_n_examples = len(translation_dataset["train"])
        needed_monolingual_n_examples = total_examples - translation_n_examples
        monolingual_dataset["train"] = monolingual_dataset["train"].select(range(needed_monolingual_n_examples))

        return translation_dataset, monolingual_dataset

    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")


class LongContextMixedTrainingMixin(TibToEngWithTibMixin):
    MAX_INPUT_LENGTH = 256
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 7  # 128

    """
    A `max_input_length` of 256 fully captures 96% of all translation train data points

    1:3 mix of translation and monolingual datasets

    Out of 1,280,000 examples, the model sees
    - 320,000 translation examples (about the total number of available translation examples)
    - 960,000 monolingual examples, where the base number of monolingual examples is about 900,000 (before applying MLM)

    This translates to
    - 27,000,000 or 27 million translation tokens for 320,000 or 320k translation pairs
        - Average of ~ 90 tokens per translation pair
    - 960,000 * 256 ~ 250,000,000 or 250 million monolingual tokens

    """
    def _create_mix(self, translation_dataset: DatasetDict, monolingual_dataset: DatasetDict) -> Tuple[DatasetDict, DatasetDict]:
        num_train_steps = self.NUM_TRANSLATION_TRAIN_STEPS
        num_examples_per_train_step = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE

        total_examples = num_train_steps * num_examples_per_train_step

        def repeat_examples(dataset: Dataset, target_n_examples: int) -> Dataset:
            if len(dataset) >= target_n_examples:
                return dataset.select(range(target_n_examples))

            indices_iter = cycle(range(len(dataset)))
            indices = [next(indices_iter) for _ in range(target_n_examples)]

            return dataset.select(indices)

        translation_dataset["train"] = repeat_examples(translation_dataset["train"], total_examples // 4)
        monolingual_dataset["train"] = repeat_examples(monolingual_dataset["train"], 3 * total_examples // 4)

        return translation_dataset, monolingual_dataset


class LC_MT_v2_Mixin(TibToEngWithTibMixin):
    MAX_INPUT_LENGTH = 256
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10  # 1024

    """
    LC_MT stands for Long Context with Mixed Training

    A `max_input_length` of 256 fully captures 96% of all translation train data points

    1:3 mix of translation and monolingual datasets

    Number of available:
    - translation examples: 320k
    - monolingual tokens: 250 mil

    Out of 10 mil examples, the model sees
    - 2.5 mil translation examples (about the 8 times total number of available translation examples)
    - 7.5 mil monolingual examples, where the base number of monolingual examples is about 900,000 (before applying MLM)

    This translates to
    - 220 mil translation tokens
        - Average of 90 tokens per translation pair
    - 7.5 mil * 256 = 2 billion monolingual tokens

    """
    def _create_mix(self, translation_dataset: DatasetDict, monolingual_dataset: DatasetDict) -> Tuple[DatasetDict, DatasetDict]:
        num_train_steps = self.NUM_TRANSLATION_TRAIN_STEPS
        num_examples_per_train_step = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE

        total_examples = num_train_steps * num_examples_per_train_step

        def repeat_examples(dataset: Dataset, target_n_examples: int) -> Dataset:
            if len(dataset) >= target_n_examples:
                return dataset.select(range(target_n_examples))

            indices_iter = cycle(range(len(dataset)))
            indices = [next(indices_iter) for _ in range(target_n_examples)]

            return dataset.select(indices)

        translation_dataset["train"] = repeat_examples(translation_dataset["train"], total_examples // 4)
        monolingual_dataset["train"] = repeat_examples(monolingual_dataset["train"], 3 * total_examples // 4)

        return translation_dataset, monolingual_dataset
