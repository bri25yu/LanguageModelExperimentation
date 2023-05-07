from typing import List, Tuple

from os import listdir
from os.path import abspath, dirname, exists, isdir, join

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from numpy import array

import matplotlib.pyplot as plt


ROOT_DIR = dirname(abspath(__file__))
RESULTS_DIR = join(ROOT_DIR, "..", "results")
GRAPHS_DIR = join(ROOT_DIR, "..", "graphs")


__all__ = ["plot_experiment", "plot_comparative_experiment"]


def get_subdirs(path: str) -> List[Tuple[str, str]]:
    """
    Returns (subdir, subdir_path).
    """
    subs = listdir(path)
    return [(s, join(path, s)) for s in subs if isdir(join(path, s))]


def get_only_subdir(path: str) -> str:
    subdirs = get_subdirs(path)
    assert len(subdirs) == 1, f"`get_only_subdir` expects a single subdirectory, but found {len(subdirs)} subdirectories at {path}"
    return subdirs[0]


def get_property_values(path: str, property_name: str) -> Tuple[List[float], List[float]]:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars(property_name)

    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    return steps, values


def get_log_dirs(experiment_name: str) -> List[Tuple[str, str]]:
    experiment_dir = join(RESULTS_DIR, experiment_name)

    res = []
    for subdir, subdir_path in get_subdirs(experiment_dir):
        subdir_path = join(subdir_path, "runs")
        subdir_path = get_only_subdir(subdir_path)[1]
        res.append((subdir, subdir_path))

    return res


def plot_experiment(experiment_name: str, title: str, property_name: str, y_label: str) -> None:
    """
    Plots the experiments by learning rate over time

    Parameters
    ----------
    experiment_name: str
        Name of an experiment like TranslationMT5600MExperiment.
    title: str
        Title of the plot.
    property_name: str
        Property to plot over time.
    y_label: str
        The label for the y-axis.

    """
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for subdir, subdir_path in get_log_dirs(experiment_name):
        steps, values = get_property_values(subdir_path, property_name)
        ax.plot(steps, values, label=subdir)


    ax.set_xlabel("Steps")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(join(GRAPHS_DIR, experiment_name))


def plot_comparative_experiment(
    experiment_names: List[str],
    legend_labels: List[str],
    title: str,
    property_name: str,
    y_label: str,
    save_name: str,
    y_min: float=None,
) -> None:
    if not save_name.endswith(".png"): save_name += ".png"
    save_path = join(GRAPHS_DIR, save_name)
    if exists(save_path):
        print(f"Already have a graph at {save_path}")
        return

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for experiment_name, legend_label in zip(experiment_names, legend_labels):
        data = []
        for _, subdir_path in get_log_dirs(experiment_name):
            steps, values = get_property_values(subdir_path, property_name)
            data.append(values)

        data = array(data)
        stds = data.std(axis=0)
        means = data.mean(axis=0)

        ax.plot(steps, means, label=legend_label)
        ax.fill_between(steps, means-stds, means+stds, alpha=.1)

    if y_min is not None:
        y_min = min(y_min, ax.get_ylim()[0])
        y_max = ax.get_ylim()[1]
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Steps")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)


if __name__ == "__main__":
    pass
