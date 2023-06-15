from lme.data_processors.flores200 import Packed2UnalignedDataProcessor

from lme.experiments.flores_300m_exps.packed2 import FloresPacked2ExperimentBase


class FloresPacked2UnalignedExperimentBase(FloresPacked2ExperimentBase):
    DATA_PROCESSOR_CLS = Packed2UnalignedDataProcessor


class FloresPacked2Unaligned300MExperiment(FloresPacked2UnalignedExperimentBase):
    pass
