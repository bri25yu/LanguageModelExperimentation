"""
A randomly selected set of pretrain span corruption examples with some baseline
translation train examples.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 2560000
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import create_mix


RATIO_PRETRAIN = 0.8  # 80% pretrain, 20% translation
TOTAL_EXAMPLES = 2560000
SEED = 42
DATASET_NAME = "flores200_pretrain_packed_mix_mt5"


def main():
    pretrain_packed_dataset = load_dataset("bri25yu/flores200_pretrain_packed_mt5")["train"]
    baseline_packed_dataset = load_dataset("bri25yu/flores200_packing")["train"]

    mixed_dataset = create_mix(pretrain_packed_dataset, baseline_packed_dataset, TOTAL_EXAMPLES, RATIO_PRETRAIN, seed=SEED)
    print(f"Mixed dataset\n{mixed_dataset}\n{mixed_dataset[0]}")

    dataset_dict = DatasetDict({
        "train": mixed_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test"),
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
