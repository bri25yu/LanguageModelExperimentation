"""
A randomly selected set of training examples tokenized with the mT5 tokenizer.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 5120000
    })
    val: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 10000
    })
})

"""

from datasets import DatasetDict, load_dataset

from transformers import AutoTokenizer


MAX_SEQ_LEN = 128
DATASET_NAME = "flores200_baseline_small_mt5"


def main():
    train_dataset = load_dataset("bri25yu/flores200_baseline_small")["train"]
    val_test_dict = load_dataset("bri25yu/flores200_val_test")
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_test_dict["val"],
        "test": val_test_dict["test"],
    })

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        inputs = [
            f"{source_lang} {sep} {target_lang} {sep} {s}"
            for source_lang, target_lang, s in
            zip(examples["source_lang"], examples["target_lang"], examples["source"])
        ]

        model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LEN, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=MAX_SEQ_LEN, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = set(dataset_dict["train"].column_names) - set(["id"])
    dataset_dict = dataset_dict.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing"
    )

    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
