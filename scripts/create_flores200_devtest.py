from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_all, tokenize_baseline_mt5


MAX_SEQ_LEN = 1024

test_dataset = load_dataset("facebook/flores", "all")["devtest"]
test_translation_dataset = select_all(test_dataset, seed=None)
test_translation_dataset_dict = DatasetDict({"devtest": test_translation_dataset})
test_translation_dataset_dict.push_to_hub("flores200_devtest_translation_pairs")

tokenized_test_translation_dataset = tokenize_baseline_mt5(DatasetDict({"train": test_translation_dataset}), MAX_SEQ_LEN)["train"]
tokenized_test_translation_dataset_dict = DatasetDict({"devtest": tokenized_test_translation_dataset})
tokenized_test_translation_dataset_dict.push_to_hub("flores200_devtest_translation_pairs_mt5")
