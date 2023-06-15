from lme.data_processors.flores200 import Packed2MixDataProcessor

from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin

from lme.experiments.flores.baseline import FloresStagedExperimentBase


class FloresPackedExperimentBase(FloresStagedExperimentBase):
    # (2048 / 2) = 1024 // (2 ** 11 / 2 ** 1) = 2 ** 10
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10

    DATA_PROCESSOR_CLASSES = [Packed2MixDataProcessor]


class FloresPacked600MExperiment(MT5600MModelMixin, FloresPackedExperimentBase):
    pass


class FloresPacked1BExperiment(MT51BModelMixin, FloresPackedExperimentBase):
    pass


class FloresPacked3BExperiment(MT53BModelMixin, FloresPackedExperimentBase):
    pass
