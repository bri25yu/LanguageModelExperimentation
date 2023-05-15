import lme  # Redirect the cache

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# MODEL_NAME_LIST = ["google/mt5-xl", "google/mt5-xl", "google/mt5-large", "google/mt5-large", "google/mt5-large", "google/mt5-base", "google/mt5-base", "google/mt5-base"]
# PUSH_TO_HUB_NAME_LIST = ["mt5-3B-flores200-packed", "mt5-3B-flores200-scaffold", "mt5-1B-flores200-baseline","mt5-1B-flores200-packed", "mt5-1B-flores200-scaffold", "mt5-600M-flores200-baseline", "mt5-600M-flores200-packed", "mt5-600M-flores200-scaffold"]
# RESULT_DIRECTORY_LIST = ["FloresPacked3BExperiment", "FloresScaffold3BExperiment", "FloresBaseline1BExperiment", "FloresPacked1BExperiment", "FloresScaffold1BExperiment", "FloresBaseline600MExperiment", "FloresPacked600MExperiment", "FloresScaffold600MExperiment"]
# LR_LIST = ["1e-04", "1e-04", "1e-04", "1e-04", "1e-04", "2e-04", "3e-04", "5e-04"]
# CHECKPOINT_LIST = ["2200", "4000", "2800", "6000", "9800", "9400", "7200", "9200"]

MODEL_NAME_LIST = ["google/mt5-large", "google/mt5-base", "google/mt5-base", "google/mt5-base"]
PUSH_TO_HUB_NAME_LIST = ["mt5-1B-flores200-scaffold", "mt5-600M-flores200-baseline", "mt5-600M-flores200-packed", "mt5-600M-flores200-scaffold"]
RESULT_DIRECTORY_LIST = ["FloresScaffoldInputMix31BExperiment", "FloresBaseline600MExperiment", "FloresPacked600MExperiment", "FloresScaffoldInputMix3600MExperiment"]
LR_LIST = ["1e-04", "2e-04", "3e-04", "5e-04"]
CHECKPOINT_LIST = ["9800", "9400", "7200", "9200"]

for MODEL_NAME, PUSH_TO_HUB_NAME, RESULT_DIRECTORY, LR, CHECKPOINT in zip(MODEL_NAME_LIST, PUSH_TO_HUB_NAME_LIST, RESULT_DIRECTORY_LIST, LR_LIST, CHECKPOINT_LIST):
    # Push tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.push_to_hub(PUSH_TO_HUB_NAME)

    # Push model
    model = AutoModelForSeq2SeqLM.from_pretrained(f"./{RESULT_DIRECTORY}/{LR}/checkpoint-{CHECKPOINT}")
    model.push_to_hub(PUSH_TO_HUB_NAME)
