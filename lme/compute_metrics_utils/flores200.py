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


# Remove chain of thought reasoning input prefix from evaluation target logits 
# Since the model is trained on the task with the chain of thought reasoning outputs
# in the training target, we will remove the prefix from the evaluation target logits
def get_flores_compute_metrics_cotr(tokenizer: PreTrainedTokenizerBase) -> Callable:
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Remove chain of thought reasoning input prefix
        target_sep_token_id = tokenizer.encode("<extra_id_0>")[0]
        label_start_idx = logits.index(target_sep_token_id)
        eval_target = logits[label_start_idx+1:]

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(eval_target, skip_special_tokens=True)

        chrf_metrics = chrf.compute(
            predictions=predictions,
            references=references,
            word_order=2,
        )

        return {"chrf++": chrf_metrics["score"]}

    return compute_metrics
