from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["ScaffoldingOutputDataProcessor", "ScaffoldingInputDataProcessor"]


class ScaffoldingInputDataProcessor(AbstractDataProcessor):
    """
    DatasetDict({
        train: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 10240000
        })
        val: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 10000
        })
    })
    """

    def load(self) -> DatasetDict:
        return load_dataset("bri25yu/flores200_eng_input_scaffolding_mt5", use_auth_token=True)


class ScaffoldingOutputDataProcessor(AbstractDataProcessor):
    """
    DatasetDict({
        train: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 10240000
        })
        val: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 10000
        })
    })
    """

    def load(self) -> DatasetDict:
        return load_dataset("hlillemark/flores200_eng_output_scaffolding_mt5", use_auth_token=True)
