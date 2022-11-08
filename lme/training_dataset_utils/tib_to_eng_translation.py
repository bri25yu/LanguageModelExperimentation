from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase


def tokenize_tib_to_eng_translation(translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return translation_dataset.map(
        tokenize_fn, batched=True, remove_columns=["tibetan", "english"], desc="Tokenizing tib to eng translation"
    )


# This is an exact copy of `tokenize_tib_to_eng_translation` unless specified otherwise
def tokenize_tib_to_eng_translation_with_prefix(
    translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase, prefix: str
) -> DatasetDict:
    def tokenize_fn(examples):
        ###############################
        # START add prefix
        ###############################

        # Original code:
        # model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

        inputs_to_tokenize = [f"{prefix} {e}" for e in examples["tibetan"]]
        model_inputs = tokenizer(inputs_to_tokenize, max_length=max_input_length, truncation=True)

        ###############################
        # END add prefix
        ###############################

        labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return translation_dataset.map(
        tokenize_fn, batched=True, remove_columns=["tibetan", "english"], desc=f"Tokenizing tib to eng translation with prefix `{prefix}`"
    )
