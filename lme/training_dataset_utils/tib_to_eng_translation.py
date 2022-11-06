from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase


__all__ = ["create_tib_to_eng_translation"]


def create_tib_to_eng_translation(translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = translation_dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])

    return tokenized_dataset
