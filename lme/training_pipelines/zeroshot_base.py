from abc import abstractmethod
from typing import Callable, List, Union

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.deepspeed import deepspeed_init

from transformers import TrainingArguments, Trainer

from lme.training_pipelines.experiment_base import ExperimentBase


__all__ = ["ZeroShotExperimentBase"]


class ZeroShotExperimentBase(ExperimentBase):
    LEARNING_RATE_PLACEHOLDER = 1e-100
    TRAINER_CLS: Union[type, None] = None

    @abstractmethod
    def get_training_arguments(self, batch_size: int, *args, **kwargs) -> TrainingArguments:
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

    def run(self, batch_size: int, _: List[float]) -> None:
        learning_rate_placeholder = self.LEARNING_RATE_PLACEHOLDER
        trainer_cls = self.TRAINER_CLS
        assert trainer_cls, f"Must override the `TRAINER_CLS` property of {self.name}"

        training_arguments = self.get_training_arguments(batch_size, learning_rate_placeholder)
        self.print_on_main_process_only(training_arguments, training_arguments)

        tokenizer = self.get_tokenizer()
        tokenized_dataset = self.get_tokenized_dataset(tokenizer, training_arguments)
        self.print_on_main_process_only(training_arguments, tokenized_dataset)

        trainer = trainer_cls(
            model=self.get_model(tokenizer),
            args=training_arguments,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            data_collator=self.get_data_collator(tokenizer),
            tokenizer=tokenizer,
            compute_metrics=self.get_compute_metrics(tokenizer),
        )

        self.init_deepspeed_inference(training_arguments, trainer)

        predictions = self.get_predictions(trainer, tokenized_dataset)

        self.load_and_save_predictions_dict(trainer, learning_rate_placeholder, predictions)

    def init_deepspeed_inference(self, training_arguments: TrainingArguments, trainer: Trainer) -> None:
        """
        Currently, running inference with deepspeed without first training requires zero 3. This is a workaround
        """
        if not training_arguments.deepspeed:
            return

        deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
            trainer, num_training_steps=0
        )
        trainer.model = deepspeed_engine.module
        trainer.model_wrapped = deepspeed_engine
        trainer.deepspeed = deepspeed_engine
        trainer.optimizer = optimizer
        trainer.lr_scheduler = lr_scheduler
