from lme.data_processors.flores200 import (
    ScaffoldingInputMix3DataProcessor,
)

from lme.experiments.flores_large_model.baseline_20mil import FloresBaseline1B20milExperiment

class FloresScaffoldInputMix31BExperiment(FloresBaseline1B20milExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3DataProcessor]
