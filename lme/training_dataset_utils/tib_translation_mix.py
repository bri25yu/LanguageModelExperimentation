from datasets import Dataset, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.modeling.t5_span_mlm import create_t5_mlm_data_collator
from lme.training_dataset_utils.utils import create_mix


__all__ = [
    "create_examples_proportional_mix",
    "create_mix_by_proportion",
]


def create_mixed_training(
    translation_train_set: Dataset,
    monolingual_train_set: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
) -> Dataset:
    mlm_data_collator = create_t5_mlm_data_collator(tokenizer, max_input_length)
    mlm_data_collator.return_tensors = "np"

    monolingual_collated = monolingual_train_set.map(mlm_data_collator, batched=True)

    mixed_train_dataset: Dataset = concatenate_datasets([translation_train_set, monolingual_collated])
    mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)

    return mixed_train_dataset


def create_examples_proportional_mix(
    translation_train_set: Dataset,
    monolingual_train_set: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    total_examples: int,
) -> Dataset:
    translation_n_examples = len(translation_train_set)
    needed_monolingual_n_examples = total_examples - translation_n_examples

    monolingual_train_set = monolingual_train_set.select(range(needed_monolingual_n_examples))

    return create_mixed_training(
        translation_train_set,
        monolingual_train_set,
        tokenizer,
        max_input_length,
    )


def create_mix_by_proportion(
    translation_train_set: Dataset,
    monolingual_train_set: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    total_examples: int,
    translation_proportion: float,
    monolingual_proportion: float,
) -> Dataset:
    translation_train_set, monolingual_train_set = create_mix(
        [
            (translation_train_set, translation_proportion),
            (monolingual_train_set, monolingual_proportion),
        ],
        total_examples,
    )

    return create_mixed_training(
        translation_train_set,
        monolingual_train_set,
        tokenizer,
        max_input_length,
    )
