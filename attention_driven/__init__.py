import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
TRAIN_OUTPUT_DIR = os.path.join(ROOT_DIR, "..", "train_output")
DATASET_CACHE_DIR = os.path.join(ROOT_DIR, "..", "dataset_cache")
