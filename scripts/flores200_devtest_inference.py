from typing import Dict, List

import json

from sacrebleu import CHRF
from sacrebleu.utils import sum_of_lists

import lme  # redirect cache

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


chrf = CHRF(6, 2, 2)  # character n-gram order 6, word n-gram order 2, beta 2

def chrf_unreduced_to_str(hypotheses: List[str], references: List[str]):
    stats = chrf._extract_corpus_statistics(hypotheses, [references])
    return json.dumps(sum_of_lists(stats))


def get_chrf_unreduced_str(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "source_lang":  [examples["source_lang"][0]],
        "target_lang":  [examples["target_lang"][0]],
        "chrf_unreduced": [chrf_unreduced_to_str(examples["prediction"], examples["target"])],
    }


tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer)

def run_eval(model_name: str, model_path_prefix: str, batch_size: int, n_examples: int=None):
    split = "devtest" if n_examples is None else f"devtest[:{n_examples}]"
    text_dataset = load_dataset("bri25yu/flores200_devtest_translation_pairs", split=split)
    tokenized_dataset = load_dataset("bri25yu/flores200_devtest_translation_pairs_mt5", split=split)

    model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_path_prefix}/{model_name}")
    args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        bf16=True,
        predict_with_generate=True,
        deepspeed={
            "zero_optimization": {
                "stage": 0,
            },
        }
    )
    trainer = Seq2SeqTrainer(model, args, data_collator)
    tokenized_predictions = trainer.predict(tokenized_dataset).predictions
    predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)

    text_dataset = text_dataset.add_column(f"prediction", predictions)
    # 1012 is the number of sentences in the flores200 devtest set
    text_dataset = text_dataset.map(get_chrf_unreduced_str, batched=True, batch_size=1012, num_proc=16, remove_columns=text_dataset.column_names)

    text_dataset.push_to_hub(f"flores200_devtest_{model_name}")


if __name__ == "__main__":
    model_path_prefix = "hlillemark"
    bs_600m = 128
    bs_1b = None
    bs_3b = None

    run_eval("mt5-600M-flores200-baseline", model_path_prefix, bs_600m)
    run_eval("mt5-600M-flores200-packed", model_path_prefix, bs_600m)
    run_eval("mt5-600M-flores200-scaffold", model_path_prefix, bs_600m)
    run_eval("mt5-1B-flores200-baseline", model_path_prefix, bs_1b)
    run_eval("mt5-1B-flores200-packed", model_path_prefix, bs_1b)
    run_eval("mt5-1B-flores200-scaffold", model_path_prefix, bs_1b)
    run_eval("mt5-3B-flores200-baseline", model_path_prefix, bs_3b)
    run_eval("mt5-3B-flores200-packed", model_path_prefix, bs_3b)
    run_eval("mt5-3B-flores200-scaffold", model_path_prefix, bs_3b)
