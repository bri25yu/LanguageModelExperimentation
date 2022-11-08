from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["TranslationDataProcessor"]


class TranslationDataProcessor(AbstractDataProcessor):
    """
    The original loaded dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 458569
        })
        test: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 10000
        })
    })

    The output dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 458569
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

        total_eval_examples = len(dataset["test"])

        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["test"].select(range(total_eval_examples//2)),
            "test": dataset["test"].select(range(total_eval_examples//2, total_eval_examples)),
        })

        return dataset
