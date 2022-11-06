from typing import List, Tuple

from itertools import cycle

from datasets import Dataset


__all__ = ["create_mix"]


def create_mix(datasets_and_proportions: List[Tuple[Dataset, float]], total_examples: int) -> List[Dataset]:
    def repeat_examples(dataset: Dataset, target_n_examples: int) -> Dataset:
        if len(dataset) >= target_n_examples:
            return dataset.select(range(target_n_examples))

        indices_iter = cycle(range(len(dataset)))
        indices = [next(indices_iter) for _ in range(target_n_examples)]

        return dataset.select(indices)

    return [repeat_examples(d, int(p * total_examples)) for d, p in datasets_and_proportions]
