import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATASET_CACHE_DIR = os.path.join(ROOT_DIR, "..", "dataset_cache")
HUGGINGFACE_CACHE_DIR = os.path.join(ROOT_DIR, "..", "..", "huggingface_cache")


os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HUGGINGFACE_CACHE_DIR
