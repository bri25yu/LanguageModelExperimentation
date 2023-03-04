"""
A randomly selected set of training examples tokenized with the mT5 tokenizer with
incomplete curriculum applied to the first 2000 steps.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 20480000
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

from datasets import load_dataset

from lme.training_dataset_utils.flores.utils import apply_incomplete


MAX_SEQ_LEN = 128
BATCH_SIZE_PER_UPDATE = 2048
NUM_UPDATES = 10000
DATASET_NAME = "flores200_incomplete"

total_examples = BATCH_SIZE_PER_UPDATE * NUM_UPDATES


def main():
    dataset_dict = load_dataset("bri25yu/flores200_baseline_medium_mt5")

    dataset_dict["train"] = apply_incomplete(dataset_dict["train"], MAX_SEQ_LEN, total_examples)

    dataset_dict.push_to_hub(DATASET_NAME, private=True)


if __name__ == "__main__":
    main()
