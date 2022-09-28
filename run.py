"""
This is a convenience script for running experiments
"""
import shutil

from attention_driven import TRAIN_OUTPUT_DIR
from attention_driven.experiments import available_experiments


EXPERIMENTS_TO_RUN = [  # Each element is a tuple (experiment name, batch size, learning rates: Optional[List[float]])
]


for experiment_name, batch_size, learning_rates in EXPERIMENTS_TO_RUN:
    experiment_cls = available_experiments.get(experiment_name, None)

    if experiment_cls is None:
        print(f"Experiment {experiment_name} is not recognized")
        continue

    for learning_rate in learning_rates:
        experiment = experiment_cls()

        try:
            experiment.run(batch_size, [learning_rate])
        except Exception as e:
            print(e)

        shutil.rmtree(TRAIN_OUTPUT_DIR, ignore_errors=True)
