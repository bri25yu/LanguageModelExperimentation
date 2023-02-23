from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase


def tokenize_tydiqa(dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["context"], examples["question"], max_length=max_input_length, truncation="only_first")

        labels = tokenizer(text_target=examples["answer"])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_fn, batched=True, desc="Tokenizing", remove_columns=["question", "context", "answer"]
    )
