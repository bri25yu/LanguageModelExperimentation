"""
A randomly selected set of training examples tokenized with the mT5 tokenizer.

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

from lme.training_dataset_utils.flores.utils import tokenize_eng_scaffold_mt5


MAX_SEQ_LEN = 256
DATASET_NAME = "flores200_eng_output_scaffolding_mt5"


def main():
    train_dataset = load_dataset("hlillemark/flores200_eng_scaffolding")["train"]
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test")
    })

    dataset_dict = tokenize_eng_scaffold_mt5(dataset_dict, MAX_SEQ_LEN, is_scaffold_input=False)

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
