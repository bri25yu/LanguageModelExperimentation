import os

import csv
import pandas as pd

from datasets import DatasetDict

from attention_driven import ROOT_DIR
from attention_driven.data_processors.abstract import AbstractDataProcessor


class FinetuneDataProcessor(AbstractDataProcessor):
    """
    Using data from https://github.com/Linguae-Dharmae/language-models

    This data is for finetuning on Tibetan and English translation.

    This class requires the LinguaeDharmae repository cloned to the correct location. See README.md
    """

    def load(self) -> DatasetDict:
        train_dataset = self.load_single_dataset("train.tsv.gz")
        test_dataset = self.load_single_dataset("eval.tsv.gz")

        return train_dataset, test_dataset

    def load_single_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(ROOT_DIR, "..", "..", "language-models/tib/data", path),
            sep="\t",
            quoting=csv.QUOTE_NONE,
            names=["tibetan","english"],
        )
        df = df.astype(str)

        return df
