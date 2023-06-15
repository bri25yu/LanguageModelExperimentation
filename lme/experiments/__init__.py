from typing import Dict, List, Tuple, Type

import sys
from types import ModuleType

from importlib import import_module
import pkgutil


__all__ = ["run"]


def get_experiments_from_module(module: ModuleType) -> Dict[str, Type]:
    experiments: Dict[str, Type] = dict()
    for attr_name in dir(module):
        ends_with_experiment = attr_name.endswith("Experiment")
        if not ends_with_experiment:
            continue

        value = getattr(module, attr_name)
        is_type = isinstance(value, type)
        if not is_type:
            continue

        experiment_cls = value
        experiments[experiment_cls.__name__] = experiment_cls

    return experiments


def recursively_discover_experiments(
    package: ModuleType,
) -> Dict[str, Type]:
    """
    Recursively import all submodules of the input module, and returns the
    classes which:
    1. Name ends in "Experiment" e.g. "BaselineExperiment"

    Parameters
    ----------
    package: ModuleType
        The input package to discover experiments in.
    Returns
    -------
    experiments: Dict[str, Type]
        A mapping of experiment name to experiment class.
    """
    package_name = package.__name__

    experiments = {}
    for _, child_package_name, is_package in pkgutil.walk_packages(package.__path__):
        child_package_full_name = f"{package_name}.{child_package_name}"
        child_module = import_module(child_package_full_name)

        # If the current module is a package, we need to recurse into its submodules
        if is_package:
            experiments.update(recursively_discover_experiments(child_module))

        # We update our experiments dict with experiments from this module
        experiments.update(get_experiments_from_module(child_module))

    return experiments


current_module = sys.modules[__name__]
available_experiments = recursively_discover_experiments(
    current_module
)


def run(experiment_name: str, batch_size: int, learning_rates: List[float]) -> None:
    experiment_cls = available_experiments.get(experiment_name, None)

    if experiment_cls is None:
        print(f"Experiment {experiment_name} is not recognized")
        return

    [experiment_cls().run(batch_size, [lr]) for lr in learning_rates]
