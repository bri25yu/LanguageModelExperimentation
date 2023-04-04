"""
Usage:
```bash
# Test dataset
python lme/training_dataset_utils/flores/scaffold_output_cotr_example.py --test

python lme/training_dataset_utils/flores/scaffold_output_cotr_example.py
```

A randomly selected set of scaffold output examples with eval being performed on
any tokens after an output <extra_id_0> token.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 10240000
    })
})

"""
from typing import Union

from argparse import ArgumentParser

from pprint import pformat

import lme  # Redirect cache

from datasets import DatasetDict, load_dataset

from transformers import AutoTokenizer

from lme.training_dataset_utils.flores.utils import select_n, tokenize_eng_scaffold_output_cotr_mt5


TOTAL_EXAMPLES = 10240000
MAX_SEQ_LEN = 256
SEED = 42
DATASET_NAME = "flores200_eng_output_scaffolding_cotr_mt5"


def create_and_upload_dataset(num_examples: int, dataset_name: Union[None, str]) -> None:
    flores200_dataset = load_dataset("facebook/flores", "all")["dev"]
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)

    eng_data = flores200_dataset["sentence_eng_Latn"]

    # Same procedure as the regular eng scaffolding stuff but without removing the eng column since 
    # we are not mixing it with the other tasks, and english to or from translation is still a task.
    eng_scaffold_dataset = select_n(flores200_dataset, num_examples, SEED, eng_data=eng_data)
    print("English scaffold dataset", eng_scaffold_dataset, pformat(eng_scaffold_dataset[0]), "\n", sep="\n")

    tokenized_dataset = tokenize_eng_scaffold_output_cotr_mt5(
        eng_scaffold_dataset, tokenizer, MAX_SEQ_LEN
    )
    print("Tokenized English scaffold dataset", tokenized_dataset, tokenized_dataset[0], "\n", sep="\n")

    example = tokenized_dataset[0]
    print("Example input", tokenizer.decode(example["input_ids"]), sep="\n")
    print("Example target", tokenizer.decode(example["labels"]), sep="\n")

    target_sep_token_id = tokenizer.encode("<extra_id_0>")[0]
    label_start_idx = example["labels"].index(target_sep_token_id)
    eval_target = example["labels"][label_start_idx+1:]
    print("Example eval target", eval_target, tokenizer.decode(eval_target), sep="\n")

    print()

    dataset_dict = DatasetDict({
        "train": tokenized_dataset,
        "val": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="val"),
        "test": load_dataset("bri25yu/flores200_baseline_medium_mt5", split="test"),
    })

    if dataset_name is not None:
        dataset_dict.push_to_hub(dataset_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        create_and_upload_dataset(1000, None)
    else:
        create_and_upload_dataset(TOTAL_EXAMPLES, DATASET_NAME)
