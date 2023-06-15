from typing import List

from datasets import load_dataset, concatenate_datasets


def load_and_combine(chunks: List[int]) -> None:
    datasets_to_cat = []
    for chunk in chunks:
        dataset = load_dataset(f"bri25yu/nllb_chunk{chunk}")
        datasets_to_cat.append(dataset)

    dataset = concatenate_datasets(datasets_to_cat).flatten_indices()
    dataset.push_to_hub("nllb")
