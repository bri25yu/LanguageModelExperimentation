from typing import Callable, Dict, Union

import os
import pickle

import csv
import pandas as pd
from datasets import Dataset, DatasetDict

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PredictionOutput

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    M2M100ForConditionalGeneration,
    PrinterCallback,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

from attention_driven import RESULTS_DIR
from attention_driven.attention_driven import (
    AttentionDrivenM2M100ForConditionalGeneration
)


class BaselineExperiment:
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    MAX_INPUT_LENGTH = 100

    # The number of samples in the val set
    VAL_SPLIT_SIZE = 1000

    LEARNING_RATES = [1e-5, 2e-5, 3e-5]

    trainer_cls: Trainer = Seq2SeqTrainer

    def get_tokenizer(self) -> PreTrainedTokenizer:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # We use a unidirectional tokenizer for now
        # from Tibetan to English
        tokenizer.src_lang = "bod_Tibt"
        tokenizer.tgt_lang = "eng_Latn"

        return tokenizer

    def load_data(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        This function assumes that https://github.com/Linguae-Dharmae/language-models
        has been cloned into the same root folder.
        """
        val_split_size = self.VAL_SPLIT_SIZE
        max_input_length = self.MAX_INPUT_LENGTH

        def load_single_dataset(path: str) -> Dataset:
            df = pd.read_csv(
                f"language-models/tib/data/{path}",
                sep="\t",
                quoting=csv.QUOTE_NONE,
                names=["tibetan","english"],
            )
            df = df.astype(str)

            dataset = Dataset.from_pandas(df)

            return dataset

        # Load our datasets from disk into HF Dataset's
        train_dataset = load_single_dataset("train_without_shorts.tsv.gz")
        train_val_dataset = train_dataset.train_test_split(val_split_size, seed=42)
        test_dataset = load_single_dataset("test.tsv.gz")

        dataset = DatasetDict(
            train=train_val_dataset["train"],
            val=train_val_dataset["test"],
            test=test_dataset,
        )
        print("Human readable dataset:", dataset)

        def tokenize_fn(examples):
            model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            # Set up the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])
        print("Model readable dataset:", tokenized_dataset)

        return tokenized_dataset

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        metric = evaluate.load("chrf")

        def compute_metrics(eval_preds):
            logits, label_ids = eval_preds
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            translations = tokenizer.batch_decode(logits, skip_special_tokens=True)

            return metric.compute(
                predictions=translations,
                references=[[r] for r in references],
                word_order=2,
            )

        return compute_metrics

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model

    @property
    def experiment_class_output_dir(self) -> str:
        exp_name = self.__class__.__name__
        return os.path.join(RESULTS_DIR, exp_name)

    @property
    def predictions_output_path(self) -> str:
        return os.path.join(
            self.experiment_class_output_dir, "predictions"
        )

    def get_training_arguments(
        self, learning_rate: float, batch_size: int
    ) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, "{learning_rate:.0e}"
        )

        return Seq2SeqTrainingArguments(
            output_dir,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=32 // batch_size,
            eval_accumulation_steps=1,
            max_steps=10000,
            warmup_ratio=0.1,
            do_train=True,
            do_eval=True,
            seed=42,
            fp16=True,
            log_level="error",
            logging_steps=1,
            predict_with_generate=True,
        )

    def run(self, batch_size: int) -> Dict[float, PredictionOutput]:
        max_input_length = self.MAX_INPUT_LENGTH
        learning_rates = self.LEARNING_RATES
        trainer_cls = self.trainer_cls

        tokenizer = self.get_tokenizer()
        tokenized_dataset = self.load_data(tokenizer)
        compute_metrics = self.get_compute_metrics(tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

        predictions_dict: Dict[float, PredictionOutput] = dict()
        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(learning_rate, batch_size)
            model = self.get_model(tokenizer)

            trainer = trainer_cls(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(2)],
            )
            trainer.remove_callback(PrinterCallback)

            trainer.train()

            predictions = trainer.predict(tokenized_dataset["test"])
            predictions_dict[learning_rate] = predictions

        # Save our predictions to disk
        with open(self.predictions_output_path, "wb") as f:
            pickle.dump(predictions, f)

        return predictions


class AttentionDrivenExperimentBase(BaselineExperiment):
    attention_driven_masking_probability: Union[None, float] = None

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        attention_driven_masking_probability = self.attention_driven_masking_probability

        model = AttentionDrivenM2M100ForConditionalGeneration.from_pretrained(
            model_name, attention_driven_masking_probability
        )

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class AttentionDrivenConfig1Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.15


class AttentionDrivenConfig2Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.05


class AttentionDrivenConfig3Experiment(AttentionDrivenExperimentBase):
    attention_driven_masking_probability = 0.25
