"""
A randomly selected set of packed2 input examples with some baseline
translation train examples. 80% packed2 and 20% baseline translation examples

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 10240000
    }),
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

from lme.training_dataset_utils.flores.utils import create_mix


RATIO_DATASET1 = 0.8  # 80% packed2, 20% translation
TOTAL_EXAMPLES = 10240000
SEED = 42
DATASET_NAME = "flores200_packed2_mix_mt5"


def main():
    packed2_dataset = load_dataset("bri25yu/flores200_packed2")["train"]
    baseline_dataset = load_dataset("bri25yu/flores200_baseline_medium_mt5")["train"]

    mixed_dataset = create_mix(packed2_dataset, baseline_dataset, TOTAL_EXAMPLES, RATIO_DATASET1, seed=SEED)
    print(f"Mixed dataset\n{mixed_dataset}\n{mixed_dataset[0]}")

    dataset_dict = DatasetDict({
        "train": mixed_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test"),
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
