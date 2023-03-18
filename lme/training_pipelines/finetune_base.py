from abc import abstractmethod
from typing import Callable, List, Union

import os

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainingArguments

from lme.data_processors.utils import dataset_summary
from lme.training_pipelines.experiment_base import ExperimentBase


__all__ = ["FinetuneExperimentBase"]


class FinetuneExperimentBase(ExperimentBase):
    TRAINER_CLS: Union[type, None] = None

    @abstractmethod
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        pass

    @abstractmethod
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        pass

    @abstractmethod
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        pass

    @abstractmethod
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
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
            self.print_on_main_process_only(training_arguments, training_arguments)

            if tokenized_dataset is None:
                tokenized_dataset = self.get_tokenized_dataset(tokenizer, training_arguments)
                self.print_on_main_process_only(training_arguments, dataset_summary(tokenized_dataset))

            model = self.get_model(tokenizer)
            trainer = trainer_cls(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            self.setup_trainer_log_callbacks(trainer)

            resume_from_checkpoint = self.resume_from_checkpoint(training_arguments)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            predictions = self.get_predictions(trainer, tokenized_dataset)

            self.load_and_save_predictions_dict(trainer, learning_rate, predictions)

    def resume_from_checkpoint(self, training_args: TrainingArguments) -> bool:
        user_flag = os.environ.get("RESUME_FROM_CHECKPOINT", False)
        if not user_flag: return False

        output_dir = training_args.output_dir
        last_checkpoint = get_last_checkpoint(output_dir)

        return last_checkpoint is not None
