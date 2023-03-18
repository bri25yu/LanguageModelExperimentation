from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["PretrainMixDataProcessor"]


class PretrainMixDataProcessor(AbstractDataProcessor):
    """
    DatasetDict({
        train: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 20480000
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
        return load_dataset("bri25yu/flores200_pretrain_mix_mt5", use_auth_token=True)
