from abc import ABC, abstractmethod

import os

from datasets import DatasetDict, load_dataset

from attention_driven import DATASET_CACHE_DIR


class AbstractDataProcessor(ABC):
    def __call__(self, verbose: bool=True) -> DatasetDict:
        if os.path.exists(self.path):
            print("Loading dataset from cache")
            dataset = load_dataset(self.path)
        else:
            print("Loading dataset from scratch")
            dataset = self.load()
            dataset.save_to_disk(self.path)

        if verbose:
            print(dataset)
            for split_name, split in dataset.items():
                print(f"Example from {split_name}")
                print(split.iloc[0])

        return dataset

    def path(self) -> str:
        name = self.__class__.__name
        path = os.path.join(DATASET_CACHE_DIR, name)

        return path

    @abstractmethod
    def load(self) -> DatasetDict:
        pass
