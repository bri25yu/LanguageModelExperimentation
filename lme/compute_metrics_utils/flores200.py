from typing import Callable, Tuple

from numpy.typing import NDArray
from numpy import cumsum, equal, roll

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


def remove_cotr_prefix_from_logits(tokenizer: PreTrainedTokenizerBase, logits: NDArray) -> NDArray:
    """
    >>> from lme.compute_metrics_utils.flores200 import remove_cotr_prefix_from_logits
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    >>> sentence = "i went to the store <extra_id_0> i bought some milk"
    >>> input_ids = tokenizer(sentence, return_tensors="np").input_ids
    >>> tokenized_output = remove_cotr_prefix_from_logits(tokenizer, input_ids)
    >>> actual_output = tokenizer.batch_decode(tokenized_output, skip_special_tokens=True)[0]
    >>> target_output = "i bought some milk"
    >>> assert actual_output == target_output, f"Target output: {target_output}\\nActual output: {actual_output}"

    """
    target_sep_token_id = 250099  # tokenizer.encode("<extra_id_0>")[0]
    label_start_idx = roll(equal(logits, target_sep_token_id), 1, axis=1)
    label_mask = cumsum(label_start_idx, axis=1)

    eval_target = logits.copy()
    eval_target[label_mask == 0] = tokenizer.pad_token_id

    return eval_target


# Remove chain of thought reasoning input prefix from evaluation target logits 
# Since the model is trained on the task with the chain of thought reasoning outputs
# in the training target, we will remove the prefix from the evaluation target logits
def get_flores_compute_metrics_cotr(tokenizer: PreTrainedTokenizerBase) -> Callable:
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds: Tuple[NDArray, NDArray]):
        """
        Parameters
        ----------
        eval_preds: Tuple of two numpy ndarray

        logits:     NDArray of shape (batch_size, sequence_length)
        label_ids:  NDArray of shape (batch_size, sequence_length)
        """
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        eval_target = remove_cotr_prefix_from_logits(tokenizer, logits)

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(eval_target, skip_special_tokens=True)

        chrf_metrics = chrf.compute(
            predictions=predictions,
            references=references,
            word_order=2,
        )

        return {"chrf++": chrf_metrics["score"]}

    return compute_metrics


if __name__ == "__main__":
    import doctest; doctest.testmod()
