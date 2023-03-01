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

from lme.training_dataset_utils.flores.utils import tokenize_baseline_mt5


MAX_SEQ_LEN = 128
DATASET_NAME = "flores200_baseline_medium_mt5"


def main():
    train_dataset = load_dataset("bri25yu/flores200_baseline_medium")["train"]
    val_test_dict = load_dataset("bri25yu/flores200_val_test")
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_test_dict["val"],
        "test": val_test_dict["test"],
    })

    dataset_dict = tokenize_baseline_mt5(dataset_dict)

    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
