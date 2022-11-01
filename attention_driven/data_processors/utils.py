from typing import Dict, Sequence, Union

import pandas as pd

from datasets import Dataset, DatasetDict


__all__ = ["convert_df_to_hf_dataset", "dataset_summary"]


def convert_df_to_hf_dataset(
    dfs: Union[pd.DataFrame, Sequence[pd.DataFrame], Dict[str, pd.DataFrame]]
) -> Union[Dataset, Sequence[Dataset], Dict[str, Dataset]]:
    if isinstance(dfs, pd.DataFrame):
        return Dataset.from_pandas(dfs)
    elif isinstance(dfs, Dict):
        return {k: convert_df_to_hf_dataset(v) for k, v in dfs.items()}
    elif isinstance(dfs, Sequence):
        return list(map(convert_df_to_hf_dataset, dfs))
    else:
        raise ValueError("Input dfs typing for conversion to HF datasets is not supported!")


def dataset_summary(dataset_dict: DatasetDict) -> str:
    summary = str(dataset_dict) + "\n"
    for split_name, split in dataset_dict.items():
        summary += f"Example from {split_name}\n{split[0]}\n"

    return summary
