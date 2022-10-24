from transformers import TrainingArguments

from attention_driven.experiments.model_mixins.mt5_model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from attention_driven.experiments.tib_to_eng_translation.tib_zh_eng_pretrain_mixin import TibZhEngPretrainExperimentMixin
from attention_driven.experiments.training_pipelines import PretrainExperimentBase


class PretrainMT5TestExperiment(MT5Base580MModelMixin, TibZhEngPretrainExperimentMixin, PretrainExperimentBase):
    """
    This is just a dummy experiment to test if the pretraining process actually works.
    """
    def get_pretrain_training_arguments(self, batch_size: int) -> TrainingArguments:
        pretrain_training_arguments = super().get_pretrain_training_arguments(batch_size)

        pretrain_training_arguments.max_steps = 10

        return pretrain_training_arguments

    def get_finetune_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        finetune_training_arguments = super().get_finetune_training_arguments(batch_size, learning_rate)

        finetune_training_arguments.max_steps = 10

        return finetune_training_arguments


class PretrainMT5Base580MExperiment(MT5Base580MModelMixin, TibZhEngPretrainExperimentMixin, PretrainExperimentBase):
    pass


class PretrainMT5Large1_2BExperiment(MT5Large1_2BModelMixin, TibZhEngPretrainExperimentMixin, PretrainExperimentBase):
    pass
