from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from datasets import DatasetDict, concatenate_datasets

from attention_driven.data_processors.pretrain import PretrainDataProcessor
from attention_driven.experiments.model_mixins.mt5_model_mixins import MT5Base580MModelMixin, MT5Large1_2BModelMixin
from attention_driven.experiments.tib_to_eng_translation.tib_zh_eng_pretrain_mixin import TibZhEngPretrainExperimentMixin
from attention_driven.experiments.training_pipelines import PretrainExperimentBase


class PretrainMT5TestExperiment(MT5Base580MModelMixin, TibZhEngPretrainExperimentMixin, PretrainExperimentBase):
    """
    This is just a dummy experiment to test if the pretraining process actually works.
    """

    # This is an exact copy of `TibZhEngPretrainExperimentMixin.get_pretrain_dataset` unless specified otherwise
    def get_pretrain_dataset(self, tokenizer: PreTrainedTokenizer, pretrain_training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset_dict = PretrainDataProcessor()(pretrain_training_arguments)

        def tokenize_fn(examples):
            return tokenizer(examples["text"], max_length=max_input_length, truncation=True)

        with pretrain_training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset_dict = dataset_dict.map(tokenize_fn, batched=True, remove_columns=["text"])
            tokenized_dataset = concatenate_datasets(list(tokenized_dataset_dict.items()))
            shuffled_tokenized_dataset = tokenized_dataset.shuffle(seed=42)

            ###############################
            # START reduce dataset size for test experiment
            ###############################

            shuffled_tokenized_dataset = shuffled_tokenized_dataset.select(range(10000))

            ###############################
            # END reduce dataset size for test experiment
            ###############################

            pretrain_dataset = DatasetDict({"train": shuffled_tokenized_dataset})

        return pretrain_dataset

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
