import lme  # Redirect the cache

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/nllb-200-3.3B"
PUSH_TO_HUB_NAME = "buddhist-nlp/nllb-3B-tib2eng"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.push_to_hub(PUSH_TO_HUB_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(".")
model.push_to_hub(PUSH_TO_HUB_NAME)
