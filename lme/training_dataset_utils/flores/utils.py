from typing import Dict, List

from tqdm.notebook import trange

from numpy import array, ndarray
from numpy.random import choice, seed as set_seed

from datasets import Dataset, concatenate_datasets


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
            map_fn, remove_columns=columns_to_remove, batched=True, fn_kwargs=fn_kwargs, with_indices=True
        )
        res.append(dataset)

    return concatenate_datasets(res).select(range(n)).flatten_indices()
