from lme.data_processors.flores200 import (
    ScaffoldingInputMixDataProcessor,
    ScaffoldingInputMix3DataProcessor,
    ScaffoldingInputMix3LargeDataProcessor
)

from lme.training_argument_mixins.flores import FloresMT5FinetuneLargeArgsMixin

from lme.experiments.flores_600m_exps.packed import FloresPacked600MExperiment


class FloresScaffoldInputMix600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMixDataProcessor]


class FloresScaffoldInputMix3600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3DataProcessor]


class FLoresScaffoldInputMix3Large600MExperiment(FloresMT5FinetuneLargeArgsMixin, FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3LargeDataProcessor]
