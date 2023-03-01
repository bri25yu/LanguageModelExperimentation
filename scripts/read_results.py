from typing import Any, Dict, List

import os

from argparse import ArgumentParser
import pickle

import pandas as pd

from lme import RESULTS_DIR


def read_results(results_dir: str):
    data: List[Dict[str, Any]] = []
    for experiment_name in os.listdir(results_dir):
        if not experiment_name.endswith("Experiment"):
            continue

        results_path = os.path.join(results_dir, experiment_name, "predictions")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "rb") as f:
            results = pickle.load(f)

        for learning_rate, result_by_lr in sorted(results.items()):
            for split_name, result in result_by_lr.items():
                if split_name == "train": continue  # We don't bother displaying results for the train split

                config_data = {
                    "name": experiment_name,
                    "lr": learning_rate,
                    "split": split_name,
                }

                for attr_name, attr_value in result.metrics.items():
                    if not (
                        attr_name.endswith("score")
                        or attr_name.endswith("exact_match")
                        or attr_name.endswith("chrf++")
                    ):
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
    if "exact_match" in df.columns: df.exact_match = df.exact_match.round(3)
    if "chrf++" in df.columns: df["chrf++"] = df["chrf++"].round(1)
    df.name = df.name.str.removesuffix("Experiment")

    value_names = list(set(["exact_match", "chrf++"]).intersection(set(df.columns)))
    for value_name in df.columns:
        if not value_name.endswith("_score"):
            continue

        new_value_name = value_name.removesuffix("_score")
        df = df.rename(columns={value_name: new_value_name})
        value_name = new_value_name

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
    columns_rordered = pd.MultiIndex.from_product([value_names, ["val", "test"]], names=df.columns.names)
    df = pd.DataFrame(df, columns=columns_rordered, index=index_reordered)

    print(df.to_string(justify="left", na_rep=""))

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    read_results(args.results_dir)
