"""
This class:
1) Trains with the first stage of training arguments
2) Switches to a succeeding set of training arguments with the same trainer
3) Trains with the second stage of training arguments

"""
from abc import abstractmethod
from typing import Callable, List

from datasets import Dataset

from transformers import TrainingArguments, Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.data_processors.utils import dataset_summary
from lme.training_pipelines.finetune_base import FinetuneExperimentBase


__all__ = ["FinetuneStagedTrainingArgsExperimentBase"]


class FinetuneStagedTrainingArgsExperimentBase(FinetuneExperimentBase):
    @abstractmethod
    def update_training_arguments(
        self, training_arguments: TrainingArguments, batch_size: int
    ) -> TrainingArguments:
        pass

    @abstractmethod
    def create_stage2_training_dataset(self, training_dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def get_stage2_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        pass

    # This is an exact copy of `FinetuneExperimentBase.run` unless specified otherwise
    def run(self, batch_size: int, learning_rates: List[float]) -> None:
        trainer_cls = self.TRAINER_CLS
        assert trainer_cls, f"Must override the `TRAINER_CLS` property of {self.name}"

        tokenizer = self.get_tokenizer()
        data_collator = self.get_data_collator(tokenizer)

        ########################################
        # START Staged training data collator
        ########################################

        stage2_data_collator = self.get_stage2_data_collator(tokenizer)

        ########################################
        # END Staged training data collator
        ########################################

        compute_metrics = self.get_compute_metrics(tokenizer)
        tokenized_dataset = None

        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(batch_size, learning_rate)
            self.print_on_main_process_only(training_arguments, training_arguments)

            if tokenized_dataset is None:
                tokenized_dataset = self.get_tokenized_dataset(tokenizer, training_arguments)
                self.print_on_main_process_only(training_arguments, dataset_summary(tokenized_dataset))

                ########################################
                # START Staged training dataset
                ########################################

                stage2_train_dataset = self.create_stage2_training_dataset(tokenized_dataset["train"])
                self.print_on_main_process_only(training_arguments, dataset_summary(stage2_train_dataset))

                ########################################
                # END Staged training dataset
                ########################################

            model = self.get_model(tokenizer)
            trainer: Trainer = trainer_cls(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            self.setup_trainer_log_callbacks(trainer)

            trainer.train()

            ########################################
            # START Staged training arguments and stage 2 run
            ########################################

            training_arguments = self.update_training_arguments(training_arguments, batch_size)
            self.print_on_main_process_only(training_arguments, training_arguments)

            trainer.args = training_arguments
            trainer.train_dataset = stage2_train_dataset
            trainer.data_collator = stage2_data_collator

            trainer.train(resume_from_checkpoint=True)

            ########################################
            # END Staged training arguments and stage 2 run
            ########################################

            predictions = self.get_predictions(trainer, tokenized_dataset)

            self.load_and_save_predictions_dict(trainer, learning_rate, predictions)
