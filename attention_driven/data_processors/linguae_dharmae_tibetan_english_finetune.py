import os

import csv
import pandas as pd

from attention_driven import ROOT_DIR
from attention_driven.data_processors.abstract import AbstractDataProcessor


class LDTibetanEnglishDataProcessor(AbstractDataProcessor):
    """
    Using data from https://github.com/Linguae-Dharmae/language-models

    This data is for finetuning on Tibetan and English translation.

    This class requires the LinguaeDharmae repository cloned to the correct location. See README.md
    """

    def __call__(self):
        def load_single_dataset(path: str) -> pd.DataFrame:
            df = pd.read_csv(
                os.path.join(ROOT_DIR, "..", "..", "language-models/tib/data", path),
                sep="\t",
                quoting=csv.QUOTE_NONE,
                names=["tibetan","english"],
            )
            df = df.astype(str)

            return df

        train_dataset = load_single_dataset("train_without_shorts.tsv.gz")
        test_dataset = load_single_dataset("test.tsv.gz")

        return train_dataset, test_dataset
