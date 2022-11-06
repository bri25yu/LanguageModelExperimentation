from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from lme.training_argument_mixins import ConstantLRFinetuneTrainingArgumentsMixin
from lme.experiments.translation_mixins import TranslationMixin


class TranslationMT5ExperimentBase(ConstantLRFinetuneTrainingArgumentsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationMT5BaseExperiment(MT5Base580MModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT5LargeExperiment(MT5Large1_2BModelMixin, TranslationMT5ExperimentBase):
    pass
