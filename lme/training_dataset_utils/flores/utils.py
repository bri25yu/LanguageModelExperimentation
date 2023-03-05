from typing import Dict, List, Sequence

from itertools import chain

from tqdm.auto import trange

from numpy import array, ndarray
from numpy.random import choice, seed as set_seed

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers import AutoTokenizer

from lme.training_dataset_utils.incomplete_utils import add_prefix_truncated_output


def select_n(raw_dataset: Dataset, n: int, seed: int, max_single_size: int=10000) -> Dataset:
    set_seed(seed)

    max_n_copies = max_single_size // len(raw_dataset)
    raw_dataset = concatenate_datasets([raw_dataset] * max_n_copies)

    is_lang_key = lambda s: s.startswith("sentence_")
    lang_keys = array(list(filter(is_lang_key, raw_dataset.column_names)))

    idxs = choice(len(lang_keys), size=(int(n * 1.1), 2))
    idxs = idxs[idxs[:, 0] != idxs[:, 1]][:n]
    assert idxs.shape == (n, 2), idxs.shape

    def map_fn(
        examples: Dict[str, List[str]],
        idxs: List[int],
        source_keys: ndarray=None,
        target_keys: ndarray=None,
    ) -> Dict[str, List[str]]:
        n_examples = len(examples["id"])
        source_keys = source_keys[idxs]
        target_keys = target_keys[idxs]
        return {
            "source_lang": [k[len("sentence_"):] for k in source_keys],
            "target_lang": [k[len("sentence_"):] for k in target_keys],
            "source": [examples[source_keys[i]][i] for i in range(n_examples)],
            "target": [examples[target_keys[i]][i] for i in range(n_examples)],
        }

    columns_to_remove = tuple(set(raw_dataset.column_names) - set(["id"]))

    res = []
    for i in trange((n // len(raw_dataset)) + 1, desc="Mapping"):
        start_i, end_i = i * len(raw_dataset), (i + 1) * len(raw_dataset)
        fn_kwargs = {
            "source_keys": lang_keys[idxs[start_i: end_i, 0]],
            "target_keys": lang_keys[idxs[start_i: end_i, 1]],
        }
        n_points = len(fn_kwargs["source_keys"])
        if len(raw_dataset) > n_points:
            dataset = raw_dataset.select(range(n_points))
        else:
            dataset = raw_dataset

        dataset = dataset.map(
            map_fn,
            remove_columns=columns_to_remove,
            batched=True,
            fn_kwargs=fn_kwargs,
            with_indices=True,
        )
        res.append(dataset)

    return concatenate_datasets(res).select(range(n)).flatten_indices()


def tokenize_baseline_mt5(dataset_dict: DatasetDict, max_seq_len: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        inputs = [
            f"{source_lang} {sep} {target_lang} {sep} {s}"
            for source_lang, target_lang, s in
            zip(examples["source_lang"], examples["target_lang"], examples["source"])
        ]

        model_inputs = tokenizer(inputs, max_length=max_seq_len, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=max_seq_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = set(dataset_dict["train"].column_names) - set(["id"])
    dataset_dict = dataset_dict.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing"
    )

    return dataset_dict


def apply_incomplete(train_dataset: Dataset, max_seq_len: int, total_examples: int, seed: int=42) -> Dataset:
    set_seed(seed)

    num_incomplete = int(total_examples * 0.2)
    def map_fn(inputs: Dict[str, Sequence]) -> Dict[str, Sequence]:
        add_prefix_truncated_output(inputs, max_seq_len)
        return inputs

    incomplete_examples = train_dataset.select(range(num_incomplete)).map(map_fn, desc="Applying incomplete")
    baseline_examples = train_dataset.select(range(num_incomplete, total_examples))

    return concatenate_datasets([incomplete_examples, baseline_examples]).flatten_indices()


def apply_packing(
    flores_train_dataset: Dataset,
    total_datapoints: int,
    max_seq_len_per_example: int,
    examples_per_pack: int,
    seed: int=42,
) -> Dataset:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    sep = tokenizer.eos_token

    is_lang_key = lambda s: s.startswith("sentence_")
    all_lang_keys = array(sorted(filter(is_lang_key, flores_train_dataset.column_names)))

    keys_to_langs = lambda keys: [k[len("sentence_"):] for k in keys]
    repeats_per_example = (total_datapoints // len(flores_train_dataset)) + 1

    def select_language_pairs(inputs: Dict[str, str]) -> Dict[str, Sequence]:
        source_sentences, target_sentences = [], []
        for _ in trange(repeats_per_example, desc="Selecting languages pairs"):
            source_lang_keys, target_lang_keys = choice(all_lang_keys, size=(2, examples_per_pack), replace=False)
            source_langs = keys_to_langs(source_lang_keys)
            target_langs = keys_to_langs(target_lang_keys)

            source_sentences.extend([
                f"{source_lang}{sep}{target_lang}{sep}{inputs[k]}"
                for k, source_lang, target_lang in zip(source_lang_keys, source_langs, target_langs)
            ])
            target_sentences.extend([inputs[k] for k in target_lang_keys])

        return {
            "source": source_sentences,
            "target": target_sentences,
        }

    columns_to_remove = set(flores_train_dataset.column_names) - set(["id"])
    text_dataset = flores_train_dataset.map(select_language_pairs, remove_columns=columns_to_remove, num_proc=4)

    def tokenize(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        model_inputs = tokenizer(examples["source"], max_length=max_seq_len_per_example, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=max_seq_len_per_example, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = set(text_dataset.column_names) - set(["id"])
    tokenized_dataset = text_dataset.map(
        tokenize, remove_columns=columns_to_remove, desc="Tokenizing", batched=True
    )

    def pack(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return {
            key: list(chain.from_iterable(examples[key]))
            for key in ["input_ids", "attention_mask", "labels"]
        }

    columns_to_remove = set(tokenized_dataset.column_names) - set(["id"])
    packed_dataset = tokenized_dataset.map(
        pack,
        remove_columns=columns_to_remove,
        desc="Packing",
        batched=True,
        batch_size=examples_per_pack,
    )

    return packed_dataset.shuffle(seed=seed).flatten_indices()
