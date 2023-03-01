from typing import Callable

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizerBase


def get_flores_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)

        chrf_metrics = chrf.compute(
            predictions=predictions,
            references=references,
            word_order=2,
        )

        return {"chrf++": chrf_metrics["score"]}


    return compute_metrics
