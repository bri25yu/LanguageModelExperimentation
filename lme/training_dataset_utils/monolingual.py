from datasets import Dataset

from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.modeling.t5_span_mlm import get_group_texts_fn


def tokenize_tibetan_monolingual(dataset: Dataset, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    group_texts = get_group_texts_fn(max_input_length)

    return dataset \
        .map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing ") \
        .map(group_texts, batched=True)


# This is an exact copy of `tokenize_tibetan_monolingual` unless specified otherwise
def tokenize_tibetan_monolingual_with_prefix(
    dataset: Dataset, max_input_length: int, tokenizer: PreTrainedTokenizerBase, prefix: str
) -> Dataset:
    def tokenize_fn(examples):
        ###############################
        # START add prefix
        ###############################

        # Original code:
        # return tokenizer(examples["text"])

        inputs_to_tokenize = [f"{prefix} {e}" for e in examples["text"]]
        return tokenizer(inputs_to_tokenize)

        ###############################
        # END add prefix
        ###############################

    group_texts = get_group_texts_fn(max_input_length)

    return dataset \
        .map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing ") \
        .map(group_texts, batched=True)
