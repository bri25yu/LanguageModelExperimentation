from lme.data_processors.flores200 import PackedCurriculumDataProcessor

from lme.experiments.flores.packed import FloresPackedExperimentBase


class FloresPackedCurriculumExperimentBase(FloresPackedExperimentBase):
    DATA_PROCESSOR_CLS = PackedCurriculumDataProcessor


class FloresPacked300MExperiment(FloresPackedCurriculumExperimentBase):
    pass
