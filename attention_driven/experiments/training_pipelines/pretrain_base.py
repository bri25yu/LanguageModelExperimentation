from typing import Callable, List, Union

from abc import abstractmethod

import os

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import WEIGHTS_NAME

from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)

from attention_driven.data_processors.utils import dataset_summary
from attention_driven.experiments.experiment_base import ExperimentBase


__all__ = ["PretrainExperimentBase"]


class PretrainExperimentBase(ExperimentBase):
    """
    This class pretrains for a set learning rate schedule and finetunes over multiple input learning rates
    """
    PRETRAIN_TRAINER_CLS: Union[type, None] = None
    FINETUNE_TRAINER_CLS: Union[type, None] = None

    @abstractmethod
    def get_pretrain_training_arguments(self, batch_size: int) -> TrainingArguments:
        pass

    @abstractmethod
    def get_finetune_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        pass

    @abstractmethod
    def get_pretrain_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_finetune_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_pretrain_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_finetune_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_pretrain_dataset(self, tokenizer: PreTrainedTokenizer, pretrain_training_arguments: TrainingArguments) -> DatasetDict:
        pass

    @abstractmethod
    def get_finetune_dataset(self, tokenizer: PreTrainedTokenizer, finetune_training_arguments: TrainingArguments) -> DatasetDict:
        pass

    def run(self, batch_size: int, finetune_learning_rates: List[float]) -> None:
        pretrain_trainer_cls = self.PRETRAIN_TRAINER_CLS
        finetune_trainer_cls = self.FINETUNE_TRAINER_CLS
        assert pretrain_trainer_cls, f"Must override the `PRETRAIN_TRAINER_CLS` property of {self.name}"
        assert finetune_trainer_cls, f"Must override the `FINETUNE_TRAINER_CLS` property of {self.name}"

        tokenizer = self.get_tokenizer()

        # Pretraining
        pretrain_training_arguments = self.get_pretrain_training_arguments(batch_size)
        self.print_on_main_process_only(pretrain_training_arguments, pretrain_training_arguments)

        pretrain_dataset = self.get_pretrain_dataset(tokenizer, pretrain_training_arguments)
        self.print_on_main_process_only(pretrain_training_arguments, dataset_summary(pretrain_dataset))
        pretrain_trainer: Trainer = pretrain_trainer_cls(
            model=self.get_model(tokenizer),
            args=pretrain_training_arguments,
            train_dataset=pretrain_dataset["train"],
            data_collator=self.get_pretrain_data_collator(tokenizer),
            tokenizer=tokenizer,
            compute_metrics=self.get_pretrain_compute_metrics(tokenizer),
        )
        self.setup_trainer_log_callbacks(pretrain_trainer)

        resume_from_checkpoint = os.path.isfile(os.path.join(pretrain_training_arguments.output_dir, WEIGHTS_NAME))
        pretrain_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        pretrained_model_checkpoint_dir = get_last_checkpoint(pretrain_training_arguments.output_dir)

        # Finetuning
        finetune_data_collator = self.get_finetune_data_collator(tokenizer)
        finetune_compute_metrics = self.get_finetune_compute_metrics(tokenizer)
        finetune_dataset = None
        for finetune_learning_rate in finetune_learning_rates:
            finetune_training_arguments = self.get_finetune_training_arguments(batch_size, finetune_learning_rate)
            self.print_on_main_process_only(finetune_training_arguments, finetune_training_arguments)

            if finetune_dataset is None:
                finetune_dataset = self.get_finetune_dataset(tokenizer, finetune_training_arguments)
                self.print_on_main_process_only(finetune_training_arguments, dataset_summary(finetune_dataset))

            finetune_trainer: Trainer = finetune_trainer_cls(
                model=self.get_model(tokenizer),
                args=finetune_training_arguments,
                train_dataset=finetune_dataset["train"],
                eval_dataset=finetune_dataset["val"],
                data_collator=finetune_data_collator,
                tokenizer=tokenizer,
                compute_metrics=finetune_compute_metrics,
                callbacks=[EarlyStoppingCallback(2)],
            )
            self.setup_trainer_log_callbacks(finetune_trainer)

            finetune_trainer.train(resume_from_checkpoint=pretrained_model_checkpoint_dir)

            predictions = self.get_predictions(finetune_trainer, finetune_dataset)

            self.load_and_save_predictions_dict(finetune_trainer, finetune_learning_rate, predictions)
