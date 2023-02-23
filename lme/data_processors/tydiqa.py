from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["TyDiQADataProcessor"]


class TyDiQADataProcessor(AbstractDataProcessor):
    """
    The output dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['context', 'question', 'answer'],
            num_rows: 49881
        })
        val: Dataset({
            features: ['context', 'question', 'answer'],
            num_rows: 5077
        })
        test: Dataset({
            features: ['context', 'question', 'answer'],
            num_rows: 5077
        })
    })

    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("tydiqa", "secondary_task")

        dataset = dataset.map(
            lambda e: {"answer": e["answers"]["text"][0]},
            desc="Flattening",
            remove_columns=["answers", "id", "title"]
        )

        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["validation"],
        })

        return dataset
