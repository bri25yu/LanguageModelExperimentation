"""
This is a convenience script for running experiments
"""
import shutil

from attention_driven import TRAIN_OUTPUT_DIR
from attention_driven.experiments import *


EXPERIMENTS_TO_RUN = [  # Each element is a tuple (experiment class, batch size, learning rates: Optional[List[float]])
]


for experiment_cls, batch_size, learning_rates in EXPERIMENTS_TO_RUN:
    for learning_rate in learning_rates:
        experiment = experiment_cls()
        experiment.run(batch_size, [learning_rate])

        shutil.rmtree(TRAIN_OUTPUT_DIR, ignore_errors=True)
