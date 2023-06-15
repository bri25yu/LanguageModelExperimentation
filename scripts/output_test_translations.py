from typing import List

import os

import pickle

from tqdm import tqdm

from datasets import Dataset

from lme import RESULTS_DIR
from lme.experiments import available_experiments
from lme.data_processors import Tib2EngDataProcessor


EXPERIMENTS_TO_OUTPUT: List[str] = [
    
]


def load_results(experiment):
    with open(experiment.predictions_output_path, "rb") as f:
        results = pickle.load(f)

    best_results, best_val_score = None, float("-inf")
    for result_by_lr in results.values():
        val_result = result_by_lr["val"]
        test_result = result_by_lr["test"]

        val_score = val_result.metrics["test_bleu_score"]
        if val_score > best_val_score:
            best_results, best_val_score = test_result, val_score

    return best_results


def decode_tokens(experiment, results):
    tokenizer = experiment.get_tokenizer()

    predictions = results.predictions
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    return predictions


def output_single(experiment_name: str) -> None:
    experiment = available_experiments[experiment_name]()
    test_dataset = Tib2EngDataProcessor().load()["test"]

    results = load_results(experiment)
    predictions = decode_tokens(experiment, results)

    test_dataset: Dataset = test_dataset.add_column("prediction", predictions)

    test_dataset.to_csv(os.path.join(RESULTS_DIR, f"{experiment.name}.tsv"), sep="\t")


if __name__ == "__main__":
    for experiment_name in tqdm(EXPERIMENTS_TO_OUTPUT, desc="Outputting test translations"):
        output_single(experiment_name)
