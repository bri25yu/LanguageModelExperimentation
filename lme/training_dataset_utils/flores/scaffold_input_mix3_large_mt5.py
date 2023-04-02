"""
A randomly selected set of scaffold input examples with some baseline
translation train examples.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 20480000
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import create_mix, tokenize_eng_scaffold_mt5

MAX_SEQ_LEN = 256
RATIO_DATASET1 = 0.8  # 80% scaffold, 20% translation
TOTAL_EXAMPLES = 20480000
SEED = 42
DATASET_NAME = "flores200_eng_input_scaffolding_mix3_large_mt5"
BASE_DATASET_NAME = "flores200_eng_input_scaffolding_large_mt5"


def main():
    train_dataset = load_dataset("hlillemark/flores200_eng_scaffolding_large")["train"]
    scaffold_dataset = DatasetDict({
        "train": train_dataset
    })

    scaffold_dataset = tokenize_eng_scaffold_mt5(scaffold_dataset, MAX_SEQ_LEN, is_scaffold_input=True)

    scaffold_dataset = DatasetDict({
        "train": scaffold_dataset["train"],
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test")
        })

    scaffold_dataset.push_to_hub(BASE_DATASET_NAME)

    baseline_dataset = load_dataset("bri25yu/flores200_baseline_medium_mt5")["train"]

    mixed_dataset = create_mix(scaffold_dataset, baseline_dataset, TOTAL_EXAMPLES, RATIO_DATASET1, seed=SEED)
    print(f"Mixed dataset\n{mixed_dataset}\n{mixed_dataset[0]}")

    dataset_dict = DatasetDict({
        "train": mixed_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test"),
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
