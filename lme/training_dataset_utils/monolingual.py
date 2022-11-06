from transformers.tokenization_utils import PreTrainedTokenizerBase

from datasets import DatasetDict, concatenate_datasets

from lme.modeling.t5_span_mlm import get_group_texts_fn


__all__ = ["create_examples_proportional_monolingual"]


def create_examples_proportional_monolingual(
    tokenizer: PreTrainedTokenizerBase, max_input_length: int, multilingual_dataset: DatasetDict
) -> DatasetDict:
    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    group_texts = get_group_texts_fn(max_input_length)

    tokenized_grouped_dataset_dict = multilingual_dataset \
        .map(tokenize_fn, batched=True, remove_columns=["text"]) \
        .map(group_texts, batched=True)

    tokenized_group_dataset = concatenate_datasets(list(tokenized_grouped_dataset_dict.values()))

    shuffled_tokenized_grouped_dataset = tokenized_group_dataset.shuffle(seed=42)

    pretrain_dataset = DatasetDict({"train": shuffled_tokenized_grouped_dataset})

    return pretrain_dataset
