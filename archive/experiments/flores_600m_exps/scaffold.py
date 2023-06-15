from typing import Callable

from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors.flores200 import (
    ScaffoldingInputMixDataProcessor,
    ScaffoldingInputMix3DataProcessor,
    ScaffoldingInputMix3LargeDataProcessor,
    ScaffoldingOutputCOTRDataProcessor
)

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics_cotr

from lme.training_argument_mixins.flores import FloresMT5FinetuneLargeArgsMixin

from lme.experiments.flores_600m_exps.packed import FloresPacked600MExperiment


class FloresScaffoldInputMix600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMixDataProcessor]


class FloresScaffoldInputMix3600MExperiment(FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3DataProcessor]


class FLoresScaffoldInputMix3Large600MExperiment(FloresMT5FinetuneLargeArgsMixin, FloresPacked600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingInputMix3LargeDataProcessor]


class FloresScaffoldOutputCOTR600MExperiment(FloresScaffoldInputMix600MExperiment):
    DATA_PROCESSOR_CLASSES = [ScaffoldingOutputCOTRDataProcessor]
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics_cotr(tokenizer)
