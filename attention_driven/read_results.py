from typing import Any, Dict, List

import os
import pickle

import pandas as pd

from attention_driven import RESULTS_DIR


def read_results():
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

                config_data = {
                    "name": experiment_name,
                    "lr": learning_rate,
                    "split": split_name,
                    "loss": loss,
                }

                for attr_name, attr_value in result.metrics.items():
                    if not attr_name.endswith("score"):
                        continue

                    attr_name = attr_name.removeprefix("test_")
                    config_data[attr_name] = attr_value

                data.append(config_data)

    if not data:
        print("No results found!")
        return

    # Create a dataframe and clean it up some
    df = pd.DataFrame(data)

    # Some display improvements
    df.lr = df.lr.map(lambda lr: f"{lr:.0e}")
    df.loss = df.loss.round(3)
    df.name = df.name.str.removesuffix("Experiment")

    value_names = ["loss"]
    for value_name in df.columns:
        if not value_name.endswith("score"):
            continue

        df[value_name] = df[value_name].round(1)
        value_names.append(value_name)

    # Create a spreadsheet
    df = df.pivot_table(
        index=["name", "lr"],
        columns="split",
        values=value_names,
        sort=False,
    )

    # Order the index and columns properly
    index_reordered = pd.MultiIndex.from_tuples(sorted(df.index.values, key=lambda t: [t[0], float(t[1])]), names=df.index.names)
    columns_rordered = pd.MultiIndex.from_product([value_names, ["train", "val", "test"]], names=df.columns.names)
    df = pd.DataFrame(df, columns=columns_rordered, index=index_reordered)

    print(df.to_string())

    return df


if __name__ == "__main__":
    read_results()
