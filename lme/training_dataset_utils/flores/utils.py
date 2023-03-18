from typing import Dict, List, Sequence

from itertools import chain

from tqdm.auto import trange

from numpy import array, ndarray
from numpy.random import choice, seed as set_seed

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers import AutoTokenizer

from lme.training_dataset_utils.incomplete_utils import add_prefix_truncated_output
from lme.training_dataset_utils.span_corrupt_utils import create_span_corrupt_inputs


def select_n(raw_dataset: Dataset, n: int, seed: int, max_single_size: int=10000) -> Dataset:
    set_seed(seed)

    max_n_copies = max_single_size // len(raw_dataset)
    raw_dataset = concatenate_datasets([raw_dataset] * max_n_copies)

    is_lang_key = lambda s: s.startswith("sentence_")
    lang_keys = array(list(filter(is_lang_key, raw_dataset.column_names)))

    idxs = choice(len(lang_keys), size=(int(n * 1.5), 2))
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


def create_inputs_from_examples(
    source_langs: List[str], target_langs: List[str], source_sentences: List[str], sep: str
) -> List[str]:
    return [
        f"{source_lang} {sep} {target_lang} {sep} {source_sentence}"
        for source_lang, target_lang, source_sentence in
        zip(source_langs, target_langs, source_sentences)
    ]


def create_pretrain_input_from_examples(langs: List[str], sentences: List[str], sep: str) -> List[str]:
    return [
        f"{lang} {sep} {sentence}"
        for lang, sentence in
        zip(langs, sentences)
    ]


def tokenize_baseline_mt5(dataset_dict: DatasetDict, max_seq_len: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        inputs = create_inputs_from_examples(examples["source_lang"], examples["target_lang"], examples["source"], sep)

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


def select_language_pairs_to_pack(
    flores_train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    total_datapoints: int,
    examples_per_datapoint: int,
    seed: int=42,
) -> Dataset:
    set_seed(seed)

    sep = tokenizer.eos_token

    is_lang_key = lambda s: s.startswith("sentence_")
    all_lang_keys = array(sorted(filter(is_lang_key, flores_train_dataset.column_names)))
    keys_to_langs = lambda keys: [k[len("sentence_"):] for k in keys]

    repeats_per_datapoint = (total_datapoints // len(flores_train_dataset)) + 1

    def select_language_pairs(inputs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        source, target = [], []
        for _ in trange(repeats_per_datapoint):
            source_lang_keys, target_lang_keys = choice(all_lang_keys, size=(2, examples_per_datapoint), replace=False)
            source_langs = keys_to_langs(source_lang_keys)
            target_langs = keys_to_langs(target_lang_keys)

            source_sentences = [inputs[source_lang_key][0] for source_lang_key in source_lang_keys]
            source_inputs = create_inputs_from_examples(source_langs, target_langs, source_sentences, sep)

            source.extend(source_inputs)
            target.extend([inputs[k][0] for k in target_lang_keys])

        return {
            "id": [inputs["id"][0]] * len(source),
            "source": source,
            "target": target,
        }

    text_dataset = flores_train_dataset.map(
        select_language_pairs,
        remove_columns=flores_train_dataset.column_names,
        num_proc=4,
        batched=True,
        batch_size=1,
        desc="Selecting language pairs",
    )
    assert len(text_dataset) == repeats_per_datapoint * len(flores_train_dataset) * examples_per_datapoint, len(text_dataset)

    return text_dataset.select(range(total_datapoints * examples_per_datapoint))


def tokenize_language_pairs_to_pack(text_dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_seq_len_per_example: int) -> Dataset:
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["source"], max_length=max_seq_len_per_example, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=max_seq_len_per_example, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = set(text_dataset.column_names) - set(["id"])
    return text_dataset.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing", num_proc=4
    )


def apply_packing(tokenized_dataset: Dataset, examples_per_pack: int, seed: int=42) -> Dataset:
    def pack(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return {
            "id": [examples["id"][0]],
            **{
                key: [list(chain.from_iterable(examples[key]))]
                for key in ["input_ids", "attention_mask", "labels"]
            },
        }

    packed_dataset = tokenized_dataset.map(
        pack,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing",
        batched=True,
        batch_size=examples_per_pack,
        num_proc=32,  # We use a large number of processes here since the operation per step is small
    )

    return packed_dataset.shuffle(seed=seed)


def select_pretrain(flores_train_dataset: Dataset, n: int, seed: int=42, max_single_size: int=100000) -> Dataset:
    set_seed(seed)

    max_n_copies = max_single_size // len(flores_train_dataset)
    flores_train_dataset = concatenate_datasets([flores_train_dataset] * max_n_copies)

    is_lang_key = lambda s: s.startswith("sentence_")
    lang_keys = array(list(filter(is_lang_key, flores_train_dataset.column_names)))

    def map_fn(examples: Dict[str, List[str]], idxs: List[int], batch_lang_keys: Sequence[str]) -> Dict[str, List[str]]:
        batch_lang_keys = batch_lang_keys[idxs]
        batch_langs = [k[len("sentence_"):] for k in batch_lang_keys]

        sentences = [examples[k][i] for i, k in enumerate(batch_lang_keys)]

        return {
            "lang": batch_langs,
            "sentences": sentences,
        }

    columns_to_remove = tuple(set(flores_train_dataset.column_names) - set(["id"]))

    res = []
    for _ in trange((n // len(flores_train_dataset)) + 1, desc="Mapping"):
        batch_lang_keys = choice(lang_keys, size=(len(flores_train_dataset),))
        fn_kwargs = {"batch_lang_keys": batch_lang_keys}
        dataset = flores_train_dataset.map(
            map_fn, remove_columns=columns_to_remove, batched=True, num_proc=4, fn_kwargs=fn_kwargs, with_indices=True
        )
        res.append(dataset)

    return concatenate_datasets(res).select(range(n)).flatten_indices()


def tokenize_pretrain(pretrain_dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_seq_len: int) -> Dataset:
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        inputs = create_pretrain_input_from_examples(examples["lang"], examples["sentences"], sep)

        return tokenizer(inputs, max_length=max_seq_len, truncation=True)

    columns_to_remove = set(pretrain_dataset.column_names) - set(["id"])
    tokenized_pretrain_dataset = pretrain_dataset.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing", num_proc=16
    )

    return tokenized_pretrain_dataset


def mask_and_create_labels_for_pretrain(tokenized_pretrain_dataset: Dataset, tokenizer: PreTrainedTokenizerBase, seed: int=42) -> Dataset:
    set_seed(seed)

    MASK_PROB = 0.15
    AVERAGE_SPAN_LENGTH = 3

    sentinel_start_id = len(tokenizer) - 1  # e.g. 250100 -> 250099

    def map_fn(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        examples["labels"] = []
        for i in len(examples["input_ids"]):
            input_ids = examples["input_ids"][i]
            corrupted_input_ids, label_ids =\
                create_span_corrupt_inputs(input_ids, MASK_PROB, AVERAGE_SPAN_LENGTH, sentinel_start_id)
            examples["input_ids"][i] = corrupted_input_ids
            examples["labels"].append(label_ids)

        return examples

    masked_dataset = tokenized_pretrain_dataset.map(map_fn, batched=True, desc="Creating pretrain objective")

    return masked_dataset
