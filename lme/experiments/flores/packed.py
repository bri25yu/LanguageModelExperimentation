from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin

from lme.experiments.flores.baseline import FloresStagedExperimentBase


class FloresPackedExperimentBase(FloresStagedExperimentBase):
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
    # (2048 / 2) = 1024 // (2 ** 11 / 2 ** 1) = 2 ** 10
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10

    DATASET_HF_PATHS = ["bri25yu/flores200_packed2_mix_mt5"]


class FloresPacked600MExperiment(MT5600MModelMixin, FloresPackedExperimentBase):
    pass


class FloresPacked1BExperiment(MT51BModelMixin, FloresPackedExperimentBase):
    pass


class FloresPacked3BExperiment(MT53BModelMixin, FloresPackedExperimentBase):
    pass
