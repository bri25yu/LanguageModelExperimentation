from typing import Callable

import os

from datasets import DatasetDict, concatenate_datasets

from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
)

from attention_driven.experiments.tib_to_eng_translation.tib_to_eng_translation_mixin import TibToEngTranslationMixin
from attention_driven.data_processors import PretrainDataProcessor
from attention_driven.modeling.t5_span_mlm import PyTorchDataCollatorForT5MLM


__all__ = ["TibZhEngPretrainExperimentMixin"]


class TibZhEngPretrainExperimentMixin(TibToEngTranslationMixin):
    PRETRAIN_LEARNING_RATE = 1e-4
    PRETRAIN_TRAINER_CLS = Seq2SeqTrainer
    NUM_PRETRAIN_STEPS = 100000
    NUM_PRETRAIN_WARMUP_STEPS = 1000
    NUM_PRETRAIN_SAVE_STEPS = 1000
    TARGET_PRETRAIN_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 10  # 1024

    def get_pretrain_training_arguments(self, batch_size: int) -> TrainingArguments:
        learning_rate = self.PRETRAIN_LEARNING_RATE
        output_dir = os.path.join(
            self.experiment_class_output_dir, "pretrain", f"{learning_rate:.0e}"
        )
        max_steps = self.NUM_PRETRAIN_STEPS
        warmup_steps = self.NUM_PRETRAIN_WARMUP_STEPS
        save_steps = self.NUM_PRETRAIN_SAVE_STEPS
        target_total_batch_size_per_update = self.TARGET_PRETRAIN_TOTAL_BATCH_SIZE_PER_UPDATE
        world_size = self.get_world_size()
        per_gpu_batch_size = batch_size

        gradient_accumulation_steps = target_total_batch_size_per_update // (per_gpu_batch_size * world_size)
        gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

        return Seq2SeqTrainingArguments(
            output_dir,
            learning_rate=learning_rate,
            save_strategy="steps",
            max_steps=max_steps,
            save_steps=save_steps,
            save_total_limit=1,
            per_device_train_batch_size=per_gpu_batch_size,
            per_device_eval_batch_size=per_gpu_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            do_train=True,
            seed=42,
            fp16=True,
            log_level="error",
            log_on_each_node=False,
            logging_steps=1,
            predict_with_generate=True,
            warmup_steps=warmup_steps,
            deepspeed=self.load_deepspeed_template_args("WarmupDecayLR"),
        )

    def get_pretrain_data_collator(self, tokenizer: PreTrainedTokenizer) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return PyTorchDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=0.15,
            mean_noise_span_length=3.0,
            input_length=max_input_length,
            target_length=max_input_length,
            pad_token_id=tokenizer.pad_token_id,
        )

    def get_pretrain_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        return None

    def get_pretrain_dataset(self, tokenizer: PreTrainedTokenizer, pretrain_training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        dataset_dict = PretrainDataProcessor()(pretrain_training_arguments)

        def tokenize_fn(examples):
            tokenizer = tokenizer(examples["text"], max_length=max_input_length, truncation=True, padding="max_length")

        with pretrain_training_arguments.main_process_first(desc="Mapping dataset"):
            tokenized_dataset_dict = dataset_dict.map(tokenize_fn, batched=True, remove_columns=["text"])
            tokenized_dataset = concatenate_datasets(list(tokenized_dataset_dict.items()))
            shuffled_tokenized_dataset = tokenized_dataset.shuffle(seed=42)
            pretrain_dataset = DatasetDict({"train": shuffled_tokenized_dataset})

        return pretrain_dataset
