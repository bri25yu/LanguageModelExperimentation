from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["ScaffoldingOutputDataProcessor", "ScaffoldingInputDataProcessor",
        "ScaffoldingOutputMixDataProcessor", "ScaffoldingInputMixDataProcessor",
        "ScaffoldingInputMix2DataProcessor", "ScaffoldingInputMix3DataProcessor"]


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


class ScaffoldingInputMixDataProcessor(AbstractDataProcessor):
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
        return load_dataset("hlillemark/flores200_eng_input_scaffolding_mix_mt5", use_auth_token=True)


class ScaffoldingOutputMixDataProcessor(AbstractDataProcessor):
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
        return load_dataset("hlillemark/flores200_eng_output_scaffolding_mix_mt5", use_auth_token=True)


class ScaffoldingInputMix2DataProcessor(AbstractDataProcessor):
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
        return load_dataset("hlillemark/flores200_eng_input_scaffolding_mix2_mt5", use_auth_token=True)


class ScaffoldingInputMix3DataProcessor(AbstractDataProcessor):
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
        return load_dataset("hlillemark/flores200_eng_input_scaffolding_mix3_mt5", use_auth_token=True)
