"""
A randomly selected set of training examples tokenized with the mT5 tokenizer with
8 examples packed into every data point in the train set. The first 2000 examples are
the same as the baseline. 

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

from datasets import DatasetDict, concatenate_datasets, load_dataset


BASELINE_FRACTION = 0.2  # 20%
DATASET_NAME = "flores200_packed_curriculum"


def main():
    baseline_dataset_dict = load_dataset("bri25yu/flores200_baseline_medium_mt5")
    baseline_train_dataset = baseline_dataset_dict["train"]
    packed_train_dataset = load_dataset("bri25yu/flores200_packing", split="train")

    total_train_examples = len(packed_train_dataset)
    baseline_train_examples = int(BASELINE_FRACTION * total_train_examples)

    train_dataset = concatenate_datasets([
        baseline_train_dataset.select(range(baseline_train_examples)),
        packed_train_dataset.select(range(baseline_train_examples, total_train_examples)),
    ]).flatten_indices()
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": baseline_dataset_dict["val"],
        "test": baseline_dataset_dict["test"],
    })
    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
