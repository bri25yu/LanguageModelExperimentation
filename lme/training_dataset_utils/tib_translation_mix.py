from itertools import cycle

from datasets import Dataset, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.modeling.t5_span_mlm import create_t5_mlm_data_collator


__all__ = [
    "create_mix_by_proportion",
]


def repeat_examples(dataset: Dataset, target_n_examples: int) -> Dataset:
    if len(dataset) >= target_n_examples:
        return dataset.select(range(target_n_examples))

    indices_iter = cycle(range(len(dataset)))
    indices = [next(indices_iter) for _ in range(target_n_examples)]

    return dataset.select(indices)


def create_mix_by_proportion(
    translation_train_set: Dataset,
    monolingual_train_set: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    total_examples: int,
    translation_proportion: float,
) -> Dataset:
    """
    Both `translation_train_set` and `monolingual_train_set` are already tokenized
    """

    # Resize our datasets, repeating examples as necessary
    monolingual_proportion = 1 - translation_proportion
    translation_train_set = repeat_examples(translation_train_set, int(translation_proportion * total_examples))
    monolingual_train_set = repeat_examples(monolingual_train_set, int(monolingual_proportion * total_examples))

    # Pre-apply our MLM data collator
    mlm_data_collator = create_t5_mlm_data_collator(tokenizer, max_input_length)
    mlm_data_collator.return_tensors = "np"
    monolingual_collated = monolingual_train_set.map(mlm_data_collator, batched=True, desc="Applying MLM")

    # Concatenate and shuffle our datasets
    mixed_train_dataset: Dataset = concatenate_datasets([translation_train_set, monolingual_collated])
    mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)

    return mixed_train_dataset
