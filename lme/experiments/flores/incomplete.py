from lme.data_processors.flores200 import IncompleteDataProcessor
from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin

from lme.experiments.flores.baseline import FloresBaselineMedium2Experiment


class FloresIncompleteExperimentBase(FloresBaselineMedium2Experiment):
    DATA_PROCESSOR_CLS = IncompleteDataProcessor


class FloresIncomplete300MExperiment(FloresIncompleteExperimentBase):
    pass


class FloresIncomplete600MExperiment(MT5600MModelMixin, FloresIncompleteExperimentBase):
    pass


class FloresIncomplete1BExperiment(MT51BModelMixin, FloresIncompleteExperimentBase):
    pass


class FloresIncomplete3BExperiment(MT53BModelMixin, FloresIncompleteExperimentBase):
    pass
