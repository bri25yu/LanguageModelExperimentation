from typing import Callable, Dict, List

import os
import json

from datasets import DatasetDict

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

from transformers import (
    DataCollatorForSeq2Seq,
    PrinterCallback,
    EarlyStoppingCallback,
    TrainingArguments,
    Seq2SeqTrainingArguments,
)

from attention_driven import CONFIG_DIR
from attention_driven.experiments.finetune_mt5 import FinetuneMT5ExperimentBase
from attention_driven.data_processors import LDTibetanEnglishDataV2Processor
from attention_driven.data_processors.utils import convert_df_to_hf_dataset


class PretrainMT5ExperimentBase(FinetuneMT5ExperimentBase):
    # This is the exact same function as `FinetuneMT5ExperimentBase.load_data` unless noted otherwise
    def load_data(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        This function assumes that https://github.com/Linguae-Dharmae/language-models
        has been cloned into the same root folder.
        """
        val_split_size = self.VAL_SPLIT_SIZE
        max_input_length = self.MAX_INPUT_LENGTH

        # Load our datasets from disk into HF Dataset's
        # !TODO Load pretraining data
        data_processor = LDTibetanEnglishDataV2Processor()

        train_dataset, test_dataset = convert_df_to_hf_dataset(data_processor())
        train_val_dataset = train_dataset.train_test_split(val_split_size, seed=42)

        dataset = DatasetDict(
            train=train_val_dataset["train"],
            val=train_val_dataset["test"],
            test=test_dataset,
        )
        print("Human readable dataset:", dataset)

        def tokenize_fn(examples):
            prefix = "translate to english: "
            tibetan_inputs = [prefix + t for t in examples["tibetan"]]

            model_inputs = tokenizer(tibetan_inputs, max_length=max_input_length, truncation=True)

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])
        print("Model readable dataset:", tokenized_dataset)

        return tokenized_dataset

    def get_pretraining_compute_metrics(self, tokenizer: PreTrainedTokenizer) -> Callable:
        # !TODO implement pretraining compute metrics
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

    def get_pretraining_arguments(
        self, learning_rate: float, batch_size: int
    ) -> TrainingArguments:
        # !TODO implement pretraining arguments
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )
        num_train_epochs = self.NUM_TRAIN_EPOCHS

        try:
            import deepspeed
            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        if has_deepspeed:
            deepspeed_args_path = os.path.join(CONFIG_DIR, "deepspeed.json")
            with open(deepspeed_args_path) as f:
                deepspeed_args = json.load(f)
        else:
            deepspeed_args = None

        eval_save_strategy = "epoch"

        return Seq2SeqTrainingArguments(
            output_dir,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            evaluation_strategy=eval_save_strategy,
            save_strategy=eval_save_strategy,
            save_total_limit=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=2 * batch_size,
            gradient_accumulation_steps=32 // batch_size,
            eval_accumulation_steps=1,
            num_train_epochs=num_train_epochs,
            warmup_ratio=0.1,
            do_train=True,
            do_eval=True,
            seed=42,
            fp16=True,
            log_level="error",
            logging_steps=1,
            predict_with_generate=True,
            deepspeed=deepspeed_args,
        )

    # This is the exact same function as `BaselineExperiment.run` unless noted otherwise
    def run(self, batch_size: int, learning_rates: List[float]) -> Dict[float, PredictionOutput]:
        max_input_length = self.MAX_INPUT_LENGTH
        trainer_cls = self.trainer_cls

        tokenizer = self.get_tokenizer()
        tokenized_dataset = self.load_data(tokenizer)
        compute_metrics = self.get_compute_metrics(tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

        # !TODO implement pretraining

        # We perform hyperparam tuning over three learning rates
        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(learning_rate, batch_size)

            print("Training with", training_arguments)
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

            if training_arguments.do_train:
                trainer.train()

            predictions = dict()
            for split_name in tokenized_dataset:
                split_preds = trainer.predict(tokenized_dataset[split_name])

                if split_name != "test":
                    # We only care about the predictions for the test set
                    split_preds = PredictionOutput(
                        None, None, split_preds.metrics
                    )

                predictions[split_name] = split_preds

            if trainer.is_world_process_zero():
                self._load_and_save_predictions_dict(learning_rate, predictions)

        return predictions
