"""
This class:
1) Trains with the first stage of training arguments
2) Switches to a succeeding set of training arguments with the same trainer
3) Trains with the second stage of training arguments
... for all the stages

"""
from abc import abstractmethod
from typing import Callable, List

from datasets import DatasetDict

from transformers import TrainingArguments, Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.data_processors.utils import dataset_summary
from lme.training_pipelines.finetune_base import FinetuneExperimentBase


__all__ = ["FinetuneStagedTrainingArgsExperimentBase"]


class FinetuneStagedTrainingArgsExperimentBase(FinetuneExperimentBase):
    @abstractmethod
    def get_tokenized_datasets(
        self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments
    ) -> List[DatasetDict]:
        pass

    @abstractmethod
    def update_training_arguments(
        self, training_arguments: TrainingArguments, batch_size: int, stage: int
    ) -> None:
        """
        Stage is 1-indexed.
        """
        pass

    @abstractmethod
    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        """
        Stage is 1-indexed.
        """
        pass

    @abstractmethod
    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        """
        Stage is 1-indexed.
        """
        pass

    # This is an exact copy of `FinetuneExperimentBase.run` unless specified otherwise
    def run(self, batch_size: int, learning_rates: List[float]) -> None:
        trainer_cls = self.TRAINER_CLS
        assert trainer_cls, f"Must override the `TRAINER_CLS` property of {self.name}"

        tokenizer = self.get_tokenizer()
        compute_metrics = self.get_compute_metrics(tokenizer)
        data_collator = self.get_data_collator(tokenizer)

        ########################################
        # START Staged training datasets
        ########################################

        # Original code:
        # tokenized_dataset = None

        tokenized_datasets = None

        ########################################
        # END Staged training datasets
        ########################################

        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(batch_size, learning_rate)

            ########################################
            # START Staged training arguments and dataset
            ########################################

            # Original code:
            # self.print_on_main_process_only(training_arguments, training_arguments)

            # if tokenized_dataset is None:
            #     tokenized_dataset = self.get_tokenized_dataset(tokenizer, training_arguments)
            #     self.print_on_main_process_only(training_arguments, dataset_summary(tokenized_dataset))

            tokenized_datasets = self.get_tokenized_datasets(tokenizer, training_arguments)
            tokenized_dataset = tokenized_datasets[0]

            ########################################
            # END Staged training arguments and dataset
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

            ########################################
            # START Staged training
            ########################################

            # Original code:
            # trainer.train()

            for stage, tokenized_dataset in enumerate(tokenized_datasets):
                stage = stage + 1  # 1-indexed

                self.update_training_arguments(training_arguments, batch_size, stage)
                self.update_model(model, stage)
                self.update_data_collator(data_collator, stage)

                trainer.train_dataset = tokenized_dataset["train"]

                self.print_on_main_process_only(
                    training_arguments, f"Dataset for stage {stage}:\n{dataset_summary(tokenized_dataset)}"
                )
                self.print_on_main_process_only(training_arguments, training_arguments)

                trainer.train(resume_from_checkpoint=(stage != 1))

            # We implicitly keep the last tokenized dataset for final evaluation

            ########################################
            # END Staged training
            ########################################

            predictions = self.get_predictions(trainer, tokenized_dataset)

            self.load_and_save_predictions_dict(trainer, learning_rate, predictions)
