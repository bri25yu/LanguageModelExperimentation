from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.experiments.mixed_training.with_prefix_mixin import (
    MixedWithPrefixProportion1Mixin,
    MixedWithPrefixProportion2Mixin,
    MixedWithPrefixProportion3Mixin,
)


class MixedMT5ExperimentBase(MT5FinetuneArgsMixin, FinetuneExperimentBase):
    pass


class MixedWithPrefixProportion1ExperimentBase(MixedWithPrefixProportion1Mixin, MixedMT5ExperimentBase):
    pass


class MixedWithPrefixProportion1MT5BaseExperiment(MT5Base580MModelMixin, MixedWithPrefixProportion1ExperimentBase):
    pass


class MixedWithPrefixProportion1MT5LargeExperiment(MT5Large1_2BModelMixin, MixedWithPrefixProportion1ExperimentBase):
    pass


class MixedWithPrefixProportion2ExperimentBase(MixedWithPrefixProportion2Mixin, MixedMT5ExperimentBase):
    pass


class MixedWithPrefixProportion2MT5BaseExperiment(MT5Base580MModelMixin, MixedWithPrefixProportion2ExperimentBase):
    pass


class MixedWithPrefixProportion2MT5LargeExperiment(MT5Large1_2BModelMixin, MixedWithPrefixProportion2ExperimentBase):
    pass


class MixedWithPrefixProportion3ExperimentBase(MixedWithPrefixProportion3Mixin, MixedMT5ExperimentBase):
    pass


class MixedWithPrefixProportion3MT5BaseExperiment(MT5Base580MModelMixin, MixedWithPrefixProportion3ExperimentBase):
    pass


class MixedWithPrefixProportion3MT5LargeExperiment(MT5Large1_2BModelMixin, MixedWithPrefixProportion3ExperimentBase):
    pass
