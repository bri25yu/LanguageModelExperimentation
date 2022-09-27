from typing import Any, Callable, Dict

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizer

from attention_driven.experiments.baseline import BaselineExperiment


class BaselineChrfBLEUExperimentBase(BaselineExperiment):
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        chrf = evaluate.load("chrf")
        bleu = evaluate.load("bleu")

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
            bleu_metrics = bleu.compute(predictions=predictions, references=references)

            def prepend_to_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                return {f"{prefix}_{k}": v for k, v in d.items()}

            return {
                **prepend_to_keys(chrf_metrics, "chrf"),
                **prepend_to_keys(bleu_metrics, "bleu"),
            }

        return compute_metrics
