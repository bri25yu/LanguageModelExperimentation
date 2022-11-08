from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.mixed_training.mixin import (
    MixedProportion1Mixin,
    MixedProportion2Mixin,
    MixedProportion3Mixin,
)


class MixedMT5ExperimentBase(MT5FinetuneArgsMixin, FinetuneExperimentBase):
    pass


class MixedProportion1ExperimentBase(MixedProportion1Mixin, MixedMT5ExperimentBase):
    pass


class MixedProportion1MT5BaseExperiment(MT5Base580MModelMixin, MixedProportion1ExperimentBase):
    pass


class MixedProportion1MT5LargeExperiment(MT5Large1_2BModelMixin, MixedProportion1ExperimentBase):
    pass


class MixedProportion2ExperimentBase(MixedProportion2Mixin, MixedMT5ExperimentBase):
    pass


class MixedProportion2MT5BaseExperiment(MT5Base580MModelMixin, MixedProportion2ExperimentBase):
    pass


class MixedProportion2MT5LargeExperiment(MT5Large1_2BModelMixin, MixedProportion2ExperimentBase):
    pass


class MixedProportion3ExperimentBase(MixedProportion3Mixin, MixedMT5ExperimentBase):
    pass


class MixedProportion3MT5BaseExperiment(MT5Base580MModelMixin, MixedProportion3ExperimentBase):
    pass


class MixedProportion3MT5LargeExperiment(MT5Large1_2BModelMixin, MixedProportion3ExperimentBase):
    pass
