from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod

import os
import pickle

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PredictionOutput
from transformers import Trainer, TrainingArguments, PrinterCallback

from lme import RESULTS_DIR, TRAIN_OUTPUT_DIR


__all__ = ["ExperimentBase"]


class ExperimentBase(ABC):
    """
    This is a base experiment class that is meant to be overridden. This class assumes virtually nothing about
    the nature of a particular experiment. It provides a few utility functions such as:
    - Final evaluation of a model in a Trainer class on a dataset
    - Saving evaluation to disk
    - Predefined output location
    - A clear pattern for getting a tokenizer and a model

    In order to use this experiment class, you must implement the `get_tokenizer`, `get_model`, and `run` methods.

    """
    MAX_INPUT_LENGTH: Union[None, int] = None

    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        pass

    @abstractmethod
    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        pass

    @abstractmethod
    def run(self, batch_size: int, learning_rates: List[float]) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def experiment_class_output_dir(self) -> str:
        return os.path.join(TRAIN_OUTPUT_DIR, self.name)

    @property
    def predictions_output_path(self) -> str:
        predictions_output_path = os.path.join(
            RESULTS_DIR, self.name, "predictions"
        )
        os.makedirs(os.path.split(predictions_output_path)[0], exist_ok=True)
        return predictions_output_path

    def get_predictions(
        self,
        trainer: Trainer,
        dataset: DatasetDict,
        split_to_keep_predictions_for: str="test",
        splits_to_cap: List[str]=["train"],
    ) -> Dict[str, PredictionOutput]:
        """
        We reduce the size of all the split names in `splits_to_cap` to the first 10k examples.
        """
        predictions = dict()
        for split_name in dataset:
            dataset_split_to_predict = dataset[split_name]
            if split_name in splits_to_cap:
                dataset_split_to_predict = dataset_split_to_predict.select(range(10000))

            split_preds = trainer.predict(dataset_split_to_predict)

            if split_name != split_to_keep_predictions_for:
                # We only care about the predictions for the test set
                split_preds = PredictionOutput(
                    None, None, split_preds.metrics
                )

            predictions[split_name] = split_preds

        return predictions

    def load_and_save_predictions_dict(self, trainer: Trainer, learning_rate: float, predictions: Dict[str, PredictionOutput]) -> None:
        if not trainer.is_world_process_zero():  # Check if this is the main process
            return

        predictions_output_path = self.predictions_output_path

        # Load predictions if they exist
        if os.path.exists(predictions_output_path):
            with open(predictions_output_path, "rb") as f:
                predictions_dict: Dict[float, PredictionOutput] = pickle.load(f)
        else:
            predictions_dict: Dict[float, PredictionOutput] = dict()

        predictions_dict[learning_rate] = predictions

        # Save our predictions to disk
        with open(predictions_output_path, "wb") as f:
            pickle.dump(predictions_dict, f)

    def print_on_main_process_only(self, training_arguments: TrainingArguments, to_print: Any) -> None:
        is_main_process = training_arguments.process_index == 0
        if not is_main_process:
            return

        print(to_print)

    def setup_trainer_log_callbacks(self, trainer: Trainer) -> None:
        trainer.remove_callback(PrinterCallback)
