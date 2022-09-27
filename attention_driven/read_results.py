from typing import Any, Dict, List

import os
import pickle

import pandas as pd

from attention_driven import RESULTS_DIR


def main():
    data: List[Dict[str, Any]] = []
    for experiment_name in os.listdir(RESULTS_DIR):
        if not experiment_name.endswith("Experiment"):
            continue

        results_path = os.path.join(RESULTS_DIR, experiment_name, "predictions")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "rb") as f:
            results = pickle.load(f)

        for learning_rate, result_by_lr in sorted(results.items()):
            for split_name, result in result_by_lr.items():
                loss = result.metrics["test_loss"]
                score = result.metrics["test_score"]

                config_data = {
                    "name": experiment_name,
                    "lr": learning_rate,
                    "split": split_name,
                    "loss": loss,
                    "score": score,
                }

                data.append(config_data)

    if not data:
        print("No results found!")
        return

    # Create a dataframe and clean it up some
    df = pd.DataFrame(data)

    # Generate display order
    df = df.sort_values(["name", "lr"])

    # Some display improvements
    df.lr = df.lr.map(lambda lr: f"{lr:.0e}")
    df.loss = df.loss.round(3)
    df.score = df.score.round(1)
    df.name = df.name.str.removesuffix("Experiment")

    # Create a spreadsheet
    df = df.pivot_table(
        index=["name", "lr"],
        columns="split",
        values=["loss", "score"],
        sort=False,
    )

    # Order the columns properly
    columns_rordered = pd.MultiIndex.from_product([["loss", "score"], ["train", "val", "test"]], names=df.columns.names)
    df = pd.DataFrame(df, columns=columns_rordered)

    print(df.to_string())


if __name__ == "__main__":
    main()
