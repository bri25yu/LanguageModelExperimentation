from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["Tib2EngDataProcessor"]


class Tib2EngDataProcessor(AbstractDataProcessor):
    """
    The original loaded dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 448849
        })
        validation: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
    })

    The output dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 448849
        })
        val: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
    })
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/tib_eng_bitext", use_auth_token=True)

        dataset = dataset.rename_columns({
            "input_text": "tibetan",
            "target_text": "english",
        })

        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })

        return dataset
