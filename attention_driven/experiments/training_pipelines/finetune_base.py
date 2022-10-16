from abc import abstractmethod
from typing import Callable, List, Union

from datasets import DatasetDict

import torch

from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)

from attention_driven.experiments.experiment_base import ExperimentBase


__all__ = ["FinetuneExperimentBase"]


class FinetuneExperimentBase(ExperimentBase):
    TRAINER_CLS: Union[type, None] = None

    @abstractmethod
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        pass

    @abstractmethod
    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        pass

    @abstractmethod
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        pass

    def run(self, batch_size: int, learning_rates: List[float]) -> None:
        trainer_cls = self.TRAINER_CLS
        assert trainer_cls, f"Must override the `TRAINER_CLS` property of {self.name}"

        tokenizer = self.get_tokenizer()
        data_collator = self.get_data_collator(tokenizer)
        compute_metrics = self.get_compute_metrics(tokenizer)
        tokenized_dataset = None

        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(batch_size, learning_rate)
            self.print_training_arguments(training_arguments)

            if tokenized_dataset is None:
                tokenized_dataset = self.get_tokenized_dataset(tokenizer, training_arguments)

            model = self.get_model(tokenizer)
            trainer = trainer_cls(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(2)],
            )
            self.setup_trainer_log_callbacks(trainer)

            trainer.train()

            predictions = self.get_predictions(trainer, tokenized_dataset)

            self.load_and_save_predictions_dict(trainer, learning_rate, predictions)

            # Not sure if this is necessary, but clean up after ourselves
            del model, trainer, predictions
            torch.cuda.empty_cache()
