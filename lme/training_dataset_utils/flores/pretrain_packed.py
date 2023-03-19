"""
A randomly selected set of pretraining examples tokenized with the mT5 tokenizer with
8 examples packed into every data point in the train set.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 2560000
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

from lme.training_dataset_utils.flores.utils import (
    select_languages_for_packed_pretrain,
    tokenize_pretrain,
    mask_and_create_labels_for_pretrain,
    insert_sep_for_pretrain_packing,
    apply_packing,
)


MAX_SEQ_LEN_PER_EXAMPLE = 128
NUM_EXAMPLES_PER_DATAPOINT = 8
NUM_EXAMPLES_PER_UPDATE = 2048
NUM_UPDATES = 10000
DATASET_NAME = "flores200_pretrain_packed"

assert NUM_EXAMPLES_PER_UPDATE % NUM_EXAMPLES_PER_DATAPOINT == 0
num_datapoints_per_update = NUM_EXAMPLES_PER_UPDATE // NUM_EXAMPLES_PER_DATAPOINT
total_datapoints = num_datapoints_per_update * NUM_UPDATES


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    flores_train_dataset = load_dataset("facebook/flores", "all")["dev"]

    text_dataset = select_languages_for_packed_pretrain(
        flores_train_dataset=flores_train_dataset,
        total_datapoints=total_datapoints,
        examples_per_datapoint=NUM_EXAMPLES_PER_DATAPOINT,
    )
    print(f"Text dataset of for pretraining\n{text_dataset}\n{text_dataset[0]}")

    tokenized_dataset = tokenize_pretrain(
        pretrain_dataset=text_dataset,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN_PER_EXAMPLE,
    )
    print(f"Tokenized dataset for pretraining\n{tokenized_dataset}\n{tokenized_dataset[0]}")

    corrupted_dataset = mask_and_create_labels_for_pretrain(
        tokenized_pretrain_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    corrupted_dataset = insert_sep_for_pretrain_packing(
        tokenized_dataset=corrupted_dataset,
        tokenizer=tokenizer,
    )
    print(f"Tokenized dataset with pretraining objective applied\n{corrupted_dataset}\n{corrupted_dataset[0]}")

    packed_dataset = apply_packing(
        tokenized_dataset=corrupted_dataset,
        examples_per_pack=NUM_EXAMPLES_PER_DATAPOINT,
    )
    print(f"Packed dataset\n{packed_dataset}\n{packed_dataset[0]}")

    dataset_dict = DatasetDict({
        "train": packed_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test"),
    })
    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
