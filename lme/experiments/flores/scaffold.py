from lme.data_processors.flores200 import ScaffoldingInputMix3DataProcessor
from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin

from lme.experiments.flores.baseline import FloresStagedExperimentBase


class FloresScaffoldExperimentBase(FloresStagedExperimentBase):
    # (2048 / 2) = 1024 // (2 ** 11 / 2 ** 1) = 2 ** 10
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10

    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3DataProcessor]


class FloresScaffold600MExperiment(MT5600MModelMixin, FloresScaffoldExperimentBase):
    pass


class FloresScaffoldInputMix31BExperiment(MT51BModelMixin, FloresScaffoldExperimentBase):
    pass


class FloresScaffold3BExperiment(MT53BModelMixin, FloresScaffoldExperimentBase):
    pass
