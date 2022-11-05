from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


class PretrainDataProcessor(AbstractDataProcessor):
    """
    The original loaded dataset dict is 
    DatasetDict({
        tibetan: Dataset({
            features: ['text'],
            num_rows: 10156565
        })
        english: Dataset({
            features: ['text'],
            num_rows: 2034185
        })
        chinese: Dataset({
            features: ['text'],
            num_rows: 26165641
        })
    })
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/pretraining_data", use_auth_token=True)

        return dataset
