from typing import Any, Callable, Dict

from datasets import DatasetDict, load_dataset

import evaluate

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_dataset_utils.tib_to_eng_translation import tokenize_tib_to_eng_translation


class TranslationMixin:
    """
    The original loaded dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 448849
        })
        validation: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
    })

    The output dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 448849
        })
        val: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
    })
    """

    MAX_INPUT_LENGTH = 256  # Covers 96% of the translation dataset
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
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

            # Remove the "bleu" value and call is score
            # The original bleu score is from 0 to 1, but we scale it up to 0 to 100
            bleu_metrics["score"] = bleu_metrics["bleu"] * 100
            del bleu_metrics["bleu"]

            def prepend_to_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                return {f"{prefix}_{k}": v for k, v in d.items()}

            return {
                **prepend_to_keys(chrf_metrics, "chrf"),
                **prepend_to_keys(bleu_metrics, "bleu"),
            }

        return compute_metrics

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        with training_arguments.main_process_first(desc="Loading data"):
            translation_dataset = load_dataset("buddhist-nlp/tib_eng_bitext", use_auth_token=True)
            translation_dataset = translation_dataset.rename_columns({
                "input_text": "tibetan",
                "target_text": "english",
            })
            translation_dataset = DatasetDict({
                "train": translation_dataset["train"],
                "val": translation_dataset["validation"],
                "test": translation_dataset["test"],
            })

            tokenized_dataset = tokenize_tib_to_eng_translation(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset
