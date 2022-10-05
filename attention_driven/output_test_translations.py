from argparse import ArgumentParser

import pickle

from attention_driven.experiments import available_experiments
from attention_driven.data_processors import LDTibetanEnglishDataV2Processor


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
    _, df = LDTibetanEnglishDataV2Processor()()
    results = load_results(experiment)
    predictions = decode_tokens(experiment, results)

    for i in range(5):
        print(f"*****Example translation {i+1}*****")
        print("Tibetan input:", df.tibetan.iloc[i])
        print("English translation:", df.english.iloc[i])
        print("Predicted translation:", predictions[i])
        print()

    df["predictions"] = predictions

    df.to_csv(f"predictions/{experiment.__class__.__name__}_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
