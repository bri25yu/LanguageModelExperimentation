"""
A randomly selected set of pretrain span corruption examples.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 20480000
    })
})

"""

from datasets import DatasetDict, load_dataset

from transformers import AutoTokenizer

from lme.training_dataset_utils.flores.utils import (
    tokenize_pretrain, mask_and_create_labels_for_pretrain
)


MAX_SEQ_LEN = 128
SEED = 42
DATASET_NAME = "flores200_pretrain_mt5"


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    pretrain_dataset = load_dataset("bri25yu/flores200_pretrain")["train"]
    print(f"Pretrain dataset\n{pretrain_dataset}\n{pretrain_dataset[0]}")

    tokenized_dataset = tokenize_pretrain(pretrain_dataset, tokenizer, MAX_SEQ_LEN)
    print(f"Tokenized dataset\n{tokenized_dataset}\n{tokenized_dataset[0]}")

    corrupted_dataset = mask_and_create_labels_for_pretrain(tokenized_dataset, tokenizer, seed=SEED)
    print(f"Corrupted dataset\n{corrupted_dataset}\n{corrupted_dataset[0]}")

    val_test_dict = load_dataset("bri25yu/flores200_val_test")
    dataset_dict = DatasetDict({
        "train": corrupted_dataset,
        "val": val_test_dict["val"],
        "test": val_test_dict["test"],
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
