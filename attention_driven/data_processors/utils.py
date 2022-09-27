from typing import Dict, Sequence, Union

import pandas as pd

from datasets import Dataset


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
