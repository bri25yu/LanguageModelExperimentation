from lme.data_processors.flores200 import IncompleteDataProcessor

from lme.experiments.flores_300m_exps.baseline import FloresBaselineMedium2Experiment


class FloresIncompleteExperimentBase(FloresBaselineMedium2Experiment):
    DATA_PROCESSOR_CLS = IncompleteDataProcessor


class FloresIncomplete300MExperiment(FloresIncompleteExperimentBase):
    pass
