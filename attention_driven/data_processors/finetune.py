import os

import csv
import pandas as pd

from datasets import DatasetDict

from attention_driven import ROOT_DIR
from attention_driven.data_processors.utils import convert_df_to_hf_dataset
from attention_driven.data_processors.abstract import AbstractDataProcessor


class FinetuneDataProcessor(AbstractDataProcessor):
    """
    Using data from https://github.com/Linguae-Dharmae/language-models

    This data is for finetuning on Tibetan and English translation.

    This class requires the LinguaeDharmae repository cloned to the correct location. See README.md
    """

    VAL_SPLIT_SIZE = 1000

    def load(self) -> DatasetDict:
        val_split_size = self.VAL_SPLIT_SIZE

        train_dataset = self.load_single_dataset("train.tsv.gz")
        test_dataset = self.load_single_dataset("eval.tsv.gz")

        train_dataset, test_dataset = convert_df_to_hf_dataset((train_dataset, test_dataset))

        train_val_dataset = train_dataset.train_test_split(val_split_size, seed=42)
        dataset = DatasetDict(
            train=train_val_dataset["train"],
            val=train_val_dataset["test"],
            test=test_dataset,
        )

        return dataset

    def load_single_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(ROOT_DIR, "..", "..", "language-models/tib/data", path),
            sep="\t",
            quoting=csv.QUOTE_NONE,
        )
        df = df.rename(columns={
            "input_text": "tibetan",
            "target_text": "english",
        })
        df = df.astype(str)

        return df
