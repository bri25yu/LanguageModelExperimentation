from typing import Callable

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizerBase


__all__ = ["get_exact_match_compute_metrics"]


def get_exact_match_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:
    exact_match = evaluate.load("exact_match")

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)

        return exact_match.compute(predictions=predictions, references=references)


    return compute_metrics
