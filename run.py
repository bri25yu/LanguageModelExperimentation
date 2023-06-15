"""
experiment_name: str
    The name of the experiment class to run, as a string.
batch_size: int
    The batch size to use per device.
learning_rates: List[float]]
    The learning rates to run.

Examples:
    run("TranslationMT5LargeExperiment", 16, [1e-3])

"""
from lme import run
