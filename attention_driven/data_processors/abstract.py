from abc import ABC, abstractmethod

import os

from datasets import DatasetDict, load_from_disk

from transformers import TrainingArguments

from attention_driven import DATASET_CACHE_DIR
from attention_driven.data_processors.utils import dataset_summary


class AbstractDataProcessor(ABC):
    def __call__(self, training_arguments: TrainingArguments) -> DatasetDict:
        with training_arguments.main_process_first(desc="Loading data"):
            if os.path.exists(self.path):
                dataset = load_from_disk(self.path)
            else:
                print("Loading dataset from scratch")
                dataset = self.load()
                print("Saving dataset")
                dataset.save_to_disk(self.path)

        is_main_process = training_arguments.process_index == 0
        if is_main_process:
            print(dataset_summary(dataset))

        return dataset

    @property
    def path(self) -> str:
        name = self.__class__.__name__
        path = os.path.join(DATASET_CACHE_DIR, name)

        return path

    @abstractmethod
    def load(self) -> DatasetDict:
        pass
