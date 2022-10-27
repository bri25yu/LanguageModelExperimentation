from typing import Any, Callable, Dict

import os

from datasets import DatasetDict

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from attention_driven.data_processors import FinetuneDataProcessor
from attention_driven.experiments.experiment_base import ExperimentBase


__all__ = ["TibToEngTranslationMixin", "TibToEngTranslationWithPrefixMixin"]


class TibToEngTranslationMixin(ExperimentBase):
    """
    Mixin for Tibetan to English translation finetuning
    """
    MAX_INPUT_LENGTH = 100
    NUM_TRANSLATION_TRAIN_STEPS = 10000
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 8  # 256
    NUM_TRANSLATION_EVAL_STEPS = 200
    FINETUNE_TRAINER_CLS = TRAINER_CLS = Seq2SeqTrainer

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return self.get_translation_training_arguments(batch_size, learning_rate)

    def get_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_data_collator(tokenizer)

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        return self.get_translation_dataset(tokenizer, training_arguments)

    def get_finetune_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        return self.get_translation_training_arguments(batch_size, learning_rate)

    def get_finetune_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_data_collator(tokenizer)

    def get_finetune_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return self.get_translation_compute_metrics(tokenizer)

    def get_finetune_dataset(self, tokenizer: PreTrainedTokenizer, finetune_training_arguments: TrainingArguments) -> DatasetDict:
        return self.get_translation_dataset(tokenizer, finetune_training_arguments)

    def get_translation_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )
        max_steps = self.NUM_TRANSLATION_TRAIN_STEPS
        eval_steps = self.NUM_TRANSLATION_EVAL_STEPS
        target_total_batch_size_per_update = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE
        world_size = self.get_world_size()
        per_gpu_batch_size = batch_size

        eval_save_strategy = "steps"

        gradient_accumulation_steps = target_total_batch_size_per_update // (per_gpu_batch_size * world_size)
        gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

        return Seq2SeqTrainingArguments(
            output_dir,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            evaluation_strategy=eval_save_strategy,
            save_strategy=eval_save_strategy,
            max_steps=max_steps,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=1,
            per_device_train_batch_size=per_gpu_batch_size,
            per_device_eval_batch_size=per_gpu_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=1,
            do_train=True,
            do_eval=True,
            seed=42,
            fp16=True,
            log_level="error",
            log_on_each_node=False,
            logging_steps=1,
            predict_with_generate=True,
            warmup_steps=0,
            deepspeed=self.load_deepspeed_template_args("WarmupLR"),
        )

    def get_translation_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset = FinetuneDataProcessor()(training_arguments)

        def tokenize_fn(examples):
            model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        with training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])

        return tokenized_dataset

    def get_translation_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
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

    def get_translation_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")


class TibToEngTranslationWithPrefixMixin(TibToEngTranslationMixin):
    # This is an exact copy of `TibToEngTranslationMixin.get_translation_dataset` unless specified otherwise
    def get_translation_dataset(self, tokenizer: PreTrainedTokenizer, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset = FinetuneDataProcessor()(training_arguments)

        def tokenize_fn(examples):
            ###########################
            # START add task prefix
            ###########################

            # Original code:
            # model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            prefix = "translate Tibetan to English: "
            tibetan_inputs = [prefix + t for t in examples["tibetan"]]
            model_inputs = tokenizer(tibetan_inputs, max_length=max_input_length, truncation=True)

            ###########################
            # END add task prefix
            ###########################

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        with training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])

        return tokenized_dataset
