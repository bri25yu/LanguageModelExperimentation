from datasets import Dataset

from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.modeling.t5_span_mlm import get_group_texts_fn, get_group_texts_with_prefix_fn


def tokenize_tibetan_monolingual(dataset: Dataset, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    group_texts = get_group_texts_fn(max_input_length)

    def tokenize_and_group_fn(examples):
        return group_texts(tokenizer(examples["text"]))

    return dataset.map(
        tokenize_and_group_fn, batched=True, remove_columns=["text"], desc="Tokenizing and grouping monolingual tibetan"
    )


def tokenize_tibetan_monolingual_with_prefix(
    dataset: Dataset, max_input_length: int, tokenizer: PreTrainedTokenizerBase, prefix: str
) -> Dataset:
    tokenized_prefix = tokenizer(prefix)
    group_texts_with_prefix = get_group_texts_with_prefix_fn(max_input_length, tokenized_prefix)

    def tokenize_and_group_fn(examples):
        return group_texts_with_prefix(tokenizer(examples["text"]))

    return dataset.map(
        tokenize_and_group_fn,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing and grouping monolingual tibetan with prefix `{prefix}`",
    )
