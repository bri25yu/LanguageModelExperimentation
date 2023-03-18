"""
A randomly selected set of training examples for pretraining.

DatasetDict({
    train: Dataset({
        features: ['id', 'lang', 'sentences'],
        num_rows: 20480000
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_pretrain


BATCH_SIZE_PER_UPDATE = 2048
NUM_UPDATES = 10000

SEED = 42
DATASET_NAME = "flores200_pretrain"


def main():
    total_set_size = BATCH_SIZE_PER_UPDATE * NUM_UPDATES

    raw_dataset = load_dataset("facebook/flores", "all")["dev"]
    print(f"Raw dataset\n{raw_dataset}")

    selected_pretrain_dataset = select_pretrain(raw_dataset, total_set_size, seed=SEED)
    print(f"Pretrain dataset\n{selected_pretrain_dataset}\n{selected_pretrain_dataset[0]}")

    dataset_dict = DatasetDict({
        "train": selected_pretrain_dataset,
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
