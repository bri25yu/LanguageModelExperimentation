"""
A randomly selected set of training examples tokenized with the mT5 tokenizer with
2 different/unaligned examples packed into every data point in the train set.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 10240000
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


NUM_EXAMPLES_PER_DATAPOINT = 2
NUM_EXAMPLES_PER_UPDATE = 2048
NUM_UPDATES = 10000
DATASET_NAME = "flores200_packed2_unaligned_mt5"

assert NUM_EXAMPLES_PER_UPDATE % NUM_EXAMPLES_PER_DATAPOINT == 0
num_datapoints_per_update = NUM_EXAMPLES_PER_UPDATE // NUM_EXAMPLES_PER_DATAPOINT
total_datapoints = num_datapoints_per_update * NUM_UPDATES


def main():
    tokenized_dataset = load_dataset("bri25yu/flores200_baseline_medium_mt5", split="train")
    tokenized_dataset = tokenized_dataset.select(range(total_datapoints))
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
