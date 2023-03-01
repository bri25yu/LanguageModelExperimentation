"""
A randomly selected set of training examples.

DatasetDict({
    train: Dataset({
        features: ['id', 'source_lang', 'target_lang', 'source', 'target'],
        num_rows: <number of target rows>
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_n


BATCH_SIZE_PER_UPDATE = 1024
NUM_UPDATES = 25000

SEED = 42
DATASET_NAME = "flores200_baseline_medium"


def main():
    total_set_size = BATCH_SIZE_PER_UPDATE * NUM_UPDATES

    raw_dataset = load_dataset("facebook/flores", "all")["dev"]
    dataset_dict = DatasetDict({
        "train": select_n(raw_dataset, total_set_size, SEED),
    })

    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
