from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_all, tokenize_baseline_mt5


MAX_SEQ_LEN = 1024


def sort_by_lang_pair(dataset: Dataset) -> Dataset:
    def add_translation_lang_pair(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {
            "translation_lang_pair": [
                f"{source_lang}-{target_lang}"
                for source_lang, target_lang in zip(examples["source_lang"], examples["target_lang"])
            ],
        }

    dataset = dataset.map(add_translation_lang_pair, batched=True, num_proc=4)
    dataset = dataset.sort("translation_lang_pair").flatten_indices()
    dataset = dataset.remove_columns("translation_lang_pair")

    return dataset


test_dataset = load_dataset("facebook/flores", "all")["devtest"]
test_translation_dataset = select_all(test_dataset, seed=None)
test_translation_dataset = sort_by_lang_pair(test_translation_dataset)
test_translation_dataset_dict = DatasetDict({"devtest": test_translation_dataset})
test_translation_dataset_dict.push_to_hub("flores200_devtest_translation_pairs")

tokenized_test_translation_dataset = tokenize_baseline_mt5(DatasetDict({"train": test_translation_dataset}), MAX_SEQ_LEN)["train"]
tokenized_test_translation_dataset_dict = DatasetDict({"devtest": tokenized_test_translation_dataset})
tokenized_test_translation_dataset_dict.push_to_hub("flores200_devtest_translation_pairs_mt5")
