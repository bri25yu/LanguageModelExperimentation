"""
A randomly selected set of training examples tokenized with the mT5 tokenizer with
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
    select_language_pairs_to_pack,
    tokenize_language_pairs_to_pack,
    apply_packing,
)


MAX_SEQ_LEN_PER_EXAMPLE = 128
NUM_EXAMPLES_PER_DATAPOINT = 2
NUM_EXAMPLES_PER_UPDATE = 2048
NUM_UPDATES = 10000
DATASET_NAME = "flores200_packed2"

assert NUM_EXAMPLES_PER_UPDATE % NUM_EXAMPLES_PER_DATAPOINT == 0
num_datapoints_per_update = NUM_EXAMPLES_PER_UPDATE // NUM_EXAMPLES_PER_DATAPOINT
total_datapoints = num_datapoints_per_update * NUM_UPDATES


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    flores_train_dataset = load_dataset("facebook/flores", "all")["dev"]

    text_dataset = select_language_pairs_to_pack(
        flores_train_dataset=flores_train_dataset,
        tokenizer=tokenizer,
        total_datapoints=total_datapoints,
        examples_per_datapoint=NUM_EXAMPLES_PER_DATAPOINT,
    )
    print(f"Text dataset of language pairs\n{text_dataset}\n{text_dataset[0]}")

    tokenized_dataset = tokenize_language_pairs_to_pack(
        text_dataset=text_dataset,
        tokenizer=tokenizer,
        max_seq_len_per_example=MAX_SEQ_LEN_PER_EXAMPLE,
    )
    print(f"Tokenized dataset of language pairs\n{tokenized_dataset}")

    packed_dataset = apply_packing(
        tokenized_dataset=tokenized_dataset,
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
