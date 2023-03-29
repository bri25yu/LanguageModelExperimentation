from lme.data_processors.flores200 import (
    ScaffoldingInputMixDataProcessor,
    ScaffoldingInputMix3DataProcessor,
)

from lme.experiments.flores_600m_exps.packed import FloresPacked600MExperiment


class FloresScaffoldInputMix600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMixDataProcessor]


class FloresScaffoldInputMix3600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3DataProcessor]
