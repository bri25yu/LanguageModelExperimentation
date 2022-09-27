import os

import pickle

import pandas as pd

import evaluate

from transformers import AutoTokenizer

from attention_driven import RESULTS_DIR
from attention_driven.data_processors import LDTibetanEnglishDataProcessor


EXPERIMENT_NAME = "BaselineExperiment"


def load_results():
    results_path = os.path.join(RESULTS_DIR, EXPERIMENT_NAME, "predictions")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    best_results, best_val_score = None, float("-inf")
    for result_by_lr in results.values():
        val_result = result_by_lr["val"]
        test_result = result_by_lr["test"]

        val_score = val_result.metrics["test_score"]
        if val_score > best_val_score:
            best_results, best_val_score = test_result, val_score

    return best_results


def decode_tokens(results):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    predictions = results.predictions
    label_ids = results.label_ids

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return predictions, references


def get_bleu_score(predictions, references):
    bleu = evaluate.load("bleu")

    metrics = bleu.compute(predictions=predictions, references=references)

    return metrics["bleu"] * 100


def main():
    print("Loading dataset")
    _, df = LDTibetanEnglishDataProcessor()()

    print("Loading results")
    results = load_results()

    print("Decoding output tokens")
    predictions, references = decode_tokens(results)

    for i in range(5):
        print(f"*****Example translation {i+1}*****")
        print("Tibetan input:", df.tibetan.iloc[i])
        print("English translation:", df.english.iloc[i])
        print("Predicted translation:", predictions[i])
        print()

    print("Calculating BLEU score")
    bleu_score = get_bleu_score(predictions, references)

    print(f"{EXPERIMENT_NAME} achieves a BLEU score of {bleu_score:.1f}")

    df["predictions"] = predictions

    df.to_csv(f"{EXPERIMENT_NAME}_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
