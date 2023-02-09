from itertools import cycle

from datasets import Dataset


__all__ = ["repeat_examples"]


def repeat_examples(dataset: Dataset, target_n_examples: int) -> Dataset:
    if len(dataset) >= target_n_examples:
        return dataset.select(range(target_n_examples))

    indices_iter = cycle(range(len(dataset)))
    indices = [next(indices_iter) for _ in range(target_n_examples)]

    return dataset.select(indices)
