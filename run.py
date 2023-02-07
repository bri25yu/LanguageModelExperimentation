"""
This is a convenience script for running experiments
"""
from lme.experiments import available_experiments


"""
Parameters
----------
experiment_name: str
    The name of the experiment class to run, as a string.
batch_size: int
    The batch size to use per device.
learning_rates: List[float]]
    The learning rates to run.

Examples:
    ("TranslationMT5LargeExperiment", 16, [1e-3])

"""
EXPERIMENTS_TO_RUN = [
]


for experiment_name, batch_size, learning_rates in EXPERIMENTS_TO_RUN:
    experiment_cls = available_experiments.get(experiment_name, None)

    if experiment_cls is None:
        print(f"Experiment {experiment_name} is not recognized")
        continue

    [experiment_cls().run(batch_size, [lr]) for lr in learning_rates]
