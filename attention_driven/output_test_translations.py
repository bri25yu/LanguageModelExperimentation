from argparse import ArgumentParser

import pickle

from datasets import load_from_disk

from attention_driven.experiments import available_experiments
from attention_driven.data_processors import FinetuneDataProcessor


def load_experiment():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment name to print predictions for"
    )

    args = parser.parse_args()

    return available_experiments[args.experiment]()


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


def main():
    experiment = load_experiment()
    test_dataset = load_from_disk(FinetuneDataProcessor().path)["test"]
    results = load_results(experiment)
    predictions = decode_tokens(experiment, results)

    for i in range(5):
        print(f"*****Example translation {i+1}*****")
        print("Tibetan input:", test_dataset[i]["tibetan"])
        print("English translation:", test_dataset[i]["english"])
        print("Predicted translation:", predictions[i])
        print()


if __name__ == "__main__":
    main()
