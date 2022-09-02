import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# We move train_output dir outside of the `attention_driven` and repo root folders
TRAIN_OUTPUT_DIR = os.path.join(ROOT_DIR, "..", "..", "train_output")
