import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CACHE_DIR = os.path.join(ROOT_DIR, "..", "cache")


os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


# Transformers logging level will be controlled by the training arguments
os.environ["DATASETS_VERBOSITY"] = "error"
