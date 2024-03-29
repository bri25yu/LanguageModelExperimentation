from typing import Dict, List

from math import ceil

from itertools import chain, product

from tqdm.auto import trange

from numpy import array, ndarray
from numpy.random import choice, seed as set_seed

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers import AutoTokenizer


def select_n(raw_dataset: Dataset, n: int, seed: int, max_single_size: int=10000, eng_data: List[str]=[]) -> Dataset:
    set_seed(seed)

    max_n_copies = max_single_size // len(raw_dataset)
    # Make sure we have enough data to select from indices
    raw_dataset = concatenate_datasets([raw_dataset] * max_n_copies)

    # Do the same for eng dataset if exists
    if eng_data:
        eng_dataset = eng_data * max_n_copies
        raw_dataset = raw_dataset.add_column("eng", eng_dataset)

    # Filter out language keys
    is_lang_key = lambda s: s.startswith("sentence_")
    lang_keys = array(list(filter(is_lang_key, raw_dataset.column_names)))

    # Select n random pairs of languages. We multiply n by 1.5 to make sure we have enough
    # data to select from after removing duplicates
    idxs = choice(len(lang_keys), size=(int(n * 1.5), 2))
    # Make sure we are not translating same language to itself and select n
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

        # Without special eng data input, just do source and target language
        # i indexes the randomized language list as well as the sentence list
        # and since it it is indexing into copied versions of the dataset, the 
        # sentences line up for the different languages.
        if not eng_data:
            return {
                "source_lang": [k[len("sentence_"):] for k in source_keys],
                "target_lang": [k[len("sentence_"):] for k in target_keys],
                "source": [examples[source_keys[i]][i] for i in range(n_examples)],
                "target": [examples[target_keys[i]][i] for i in range(n_examples)],
            }
        # If english data should also be processed with each example, then add 
        # eng_source to the dictionary output. 
        else:
            return {
                "source_lang": [k[len("sentence_"):] for k in source_keys],
                "target_lang": [k[len("sentence_"):] for k in target_keys],
                "source": [examples[source_keys[i]][i] for i in range(n_examples)],
                "target": [examples[target_keys[i]][i] for i in range(n_examples)],
                "eng_source": [examples["eng"][i] for i in range(n_examples)],
            }

    columns_to_remove = tuple(set(raw_dataset.column_names) - set(["id"]))

    res = []
    for i in trange((n // len(raw_dataset)) + 1, desc="Mapping"):
        # Go through dataset length once per iteration via random indexes
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


def select_all(raw_dataset: Dataset, seed: int=42) -> Dataset:
    is_lang_key = lambda s: s.startswith("sentence_")
    lang_keys = list(filter(is_lang_key, raw_dataset.column_names))

    def map_fn(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        res = {
            "source_lang": [],
            "target_lang": [],
            "source": [],
            "target": [],
        }
        for source_key, target_key in product(lang_keys, lang_keys):
            if source_key == target_key: continue

            res["source_lang"].append(source_key[len("sentence_"):])
            res["target_lang"].append(target_key[len("sentence_"):])
            res["source"].append(examples[source_key][0])
            res["target"].append(examples[target_key][0])

        res["id"] = [examples["id"][0]] * len(res["source"])
        return res

    dataset = raw_dataset.map(
        map_fn, remove_columns=raw_dataset.column_names, batched=True, batch_size=1
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed).flatten_indices()

    return dataset


def create_inputs_from_examples(
    source_langs: List[str], target_langs: List[str], source_sentences: List[str], sep: str
) -> List[str]:
    return [
        f"{source_lang} {sep} {target_lang} {sep} {source_sentence}"
        for source_lang, target_lang, source_sentence in
        zip(source_langs, target_langs, source_sentences)
    ]


def create_eng_scaffold_inputs_from_examples(
    source_langs: List[str], target_langs: List[str], source_sentences: List[str], 
    eng_sentences: List[str], is_scaffold_input: bool, sep: str
) -> List[str]:
    # Creates inputs for either english scaffolding in the input or output sentences.
    eng_lang = "eng_Latn"
    if is_scaffold_input:
        return [
            f"{source_lang} {sep} {eng_lang} {sep} {target_lang} {sep} {source_sentence} {sep} {eng_sentence}"
            for source_lang, target_lang, source_sentence, eng_sentence in
            zip(source_langs, target_langs, source_sentences, eng_sentences)
        ]
    else:
        return [
            f"{source_lang} {sep} {eng_lang} {sep} {target_lang} {sep} {source_sentence}"
            for source_lang, target_lang, source_sentence in
            zip(source_langs, target_langs, source_sentences)
        ]


def create_eng_scaffold_outputs(target_sentences: List[str], eng_sentences: List[str], sep: str) -> List[str]:
    return [
        f"{eng_sentence} {sep} {target_sentence}"
        for target_sentence, eng_sentence in
        zip(target_sentences, eng_sentences)
    ]


def tokenize_baseline_mt5(dataset_dict: DatasetDict, max_seq_len: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        inputs = create_inputs_from_examples(examples["source_lang"], examples["target_lang"], examples["source"], sep)

        model_inputs = tokenizer(inputs, max_length=max_seq_len, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=max_seq_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = set(dataset_dict["train"].column_names) - set(["id"])
    dataset_dict = dataset_dict.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing", num_proc=32
    )

    return dataset_dict


def tokenize_eng_scaffold_mt5(dataset_dict: DatasetDict, max_seq_len: int, is_scaffold_input: bool) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    sep = tokenizer.eos_token

    def tokenize_fn(examples):
        # Generate inputs, which either has the english sentence and the english language token (input) or just the english language token (output)
        inputs = create_eng_scaffold_inputs_from_examples(examples["source_lang"], examples["target_lang"], examples["source"], examples["eng_source"], is_scaffold_input, sep)
        model_inputs = tokenizer(inputs, max_length=max_seq_len, truncation=True)

        # Generate target outputs, which either has the english sentence with the target language concatenated (output),
        # or just the normal target output with the target language (input)
        if is_scaffold_input:
            labels = tokenizer(text_target=examples["target"], max_length=max_seq_len, truncation=True)
        else:
            labels = tokenizer(text_target=create_eng_scaffold_outputs(examples["target"], examples["eng_source"], sep), max_length=max_seq_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    columns_to_remove = set(dataset_dict["train"].column_names) - set(["id"])
    dataset_dict = dataset_dict.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing", num_proc=16
    )

    return dataset_dict


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
        tokenize_fn, batched=True, remove_columns=columns_to_remove, desc="Tokenizing", num_proc=16
    )


def apply_packing(tokenized_dataset: Dataset, examples_per_pack: int, seed: int=42) -> Dataset:
    original_batch_size = 1000
    inputs_per_batch = original_batch_size // examples_per_pack
    rounded_batch_size = inputs_per_batch * examples_per_pack

    def pack(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return {
            "id": examples["id"][::examples_per_pack],
            **{
                key: [list(chain.from_iterable(
                    examples[key][examples_per_pack * i: examples_per_pack * (i+1)]
                )) for i in range(inputs_per_batch)]
                for key in ["input_ids", "attention_mask", "labels"]
            },
        }

    packed_dataset = tokenized_dataset.map(
        pack,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing",
        batched=True,
        batch_size=rounded_batch_size,
        num_proc=16,
    )

    return packed_dataset.shuffle(seed=seed)


def create_mix(
    dataset1: Dataset, dataset2: Dataset, total_examples: int, ratio_dataset_1: float, seed: int=42
) -> Dataset:
    num_examples_1 = int(total_examples * ratio_dataset_1)
    num_examples_2 = total_examples - num_examples_1

    dataset1 = dataset1.select(range(num_examples_1))
    dataset2 = dataset2.select(range(num_examples_2))

    mixed_dataset = concatenate_datasets((dataset1, dataset2))
    mixed_dataset = mixed_dataset.shuffle(seed=seed).flatten_indices()

    return mixed_dataset


def select_n_for_eng_scaffold(flores200_dataset: Dataset, n_examples: int, seed: int) -> Dataset:
    set_seed(seed)

    examples_per_sentence = ceil(n_examples / len(flores200_dataset))

    english_key = "sentence_eng_Latn"
    language_keys = [c for c in flores200_dataset.column_names if c.startswith("sentence_")]
    language_keys.remove(english_key)

    def map_fn(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        res = {
            "source": [],
            "target": [],
            "english": [],
            "source_lang": [],
            "target_lang": [],
        }
        source_keys, target_keys = choice(language_keys, size=(2, examples_per_sentence))
        for source_key, target_key in zip(source_keys, target_keys):
            res["source"].append(examples[source_key][0])
            res["target"].append(examples[target_key][0])
            res["english"].append(examples[english_key][0])
            res["source_lang"].append(source_key[len("sentence_"):])
            res["target_lang"].append(target_key[len("sentence_"):])
        res["id"] = [examples["id"][0]] * examples_per_sentence
        return res

    return flores200_dataset \
        .map(map_fn, remove_columns=flores200_dataset.column_names, batched=True, batch_size=1, num_proc=16) \
        .shuffle(seed=seed) \
        .select(range(n_examples)) \
        .flatten_indices()
