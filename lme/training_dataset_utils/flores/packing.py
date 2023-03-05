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

from lme.training_dataset_utils.flores.utils import apply_packing


MAX_SEQ_LEN_PER_EXAMPLE = 128
NUM_EXAMPLES_PER_DATAPOINT = 8
NUM_EXAMPLES_PER_UPDATE = 2048
NUM_UPDATES = 10000
DATASET_NAME = "flores200_packing"

assert NUM_EXAMPLES_PER_UPDATE % NUM_EXAMPLES_PER_DATAPOINT == 0
num_datapoints_per_update = NUM_EXAMPLES_PER_UPDATE // NUM_EXAMPLES_PER_DATAPOINT
total_datapoints = num_datapoints_per_update * NUM_UPDATES


def main():
    tokenized_dataset_dict = load_dataset("bri25yu/flores200_baseline_medium_mt5")
    flores_train_dataset = load_dataset("facebook/flores", "all")["dev"]

    train_dataset = apply_packing(
        flores_train_dataset=flores_train_dataset,
        total_datapoints=total_datapoints,
        max_seq_len_per_example=MAX_SEQ_LEN_PER_EXAMPLE,
        examples_per_pack=NUM_EXAMPLES_PER_DATAPOINT,
    )

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": tokenized_dataset_dict["val"],
        "test": tokenized_dataset_dict["test"],
    })
    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
