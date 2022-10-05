from typing import Dict, List

import os

import json

from collections import OrderedDict

import torch

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PredictionOutput
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PrinterCallback,
    TrainingArguments,
    Seq2SeqTrainingArguments,
)

from attention_driven import CONFIG_DIR
from attention_driven.experiments.baseline_v2 import BaselineV2Experiment
from attention_driven.data_processors import LDTibetanEnglishDataV2Processor
from attention_driven.data_processors.utils import convert_df_to_hf_dataset
from attention_driven.modeling.mt5_fp16_utils import scale_weights_for_fp16_t5
from attention_driven.modeling.mt5_fp16 import MT5Fp16ForConditionalGeneration


__all__ = [
    "FinetuneMT5BaseExperiment",
    "FinetuneMT5LargeExperiment",
    "FinetuneMT5XLExperiment",
    "FinetuneMT5BaseV2Experiment",
    "FinetuneMT5LargeV2Experiment",
    "FinetuneMT5XLV2Experiment",
    "FinetuneMT5BaseV3Experiment",
    "FinetuneMT5LargeV3Experiment",
    "FinetuneMT5XLV3Experiment",
    "FinetuneMT5BaseFP32Experiment",
    "FinetuneMT5LargeFP32Experiment",
    "FinetuneMT5XLFP32Experiment",
]


# We use a special version fp16 capable version of MT5
class FinetuneMT5ExperimentBase(BaselineV2Experiment):
    MODEL_NAME = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        # We don't have access to bf16 capable Ampere + GPUs so we need to workaround it
        model = MT5Fp16ForConditionalGeneration.from_pretrained(model_name)
        scale_weights_for_fp16_t5(model)

        model.config.max_length = max_input_length

        return model

    # This is the exact same function as `BaselineV2Experiment.load_data` unless noted otherwise
    def load_data(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        This function assumes that https://github.com/Linguae-Dharmae/language-models
        has been cloned into the same root folder.
        """
        val_split_size = self.VAL_SPLIT_SIZE
        max_input_length = self.MAX_INPUT_LENGTH

        # Load our datasets from disk into HF Dataset's
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

            ###########################
            # START add mt5 prefix
            ###########################

            # Original code
            # model_inputs = tokenizer(examples["tibetan"], max_length=max_input_length, truncation=True)

            prefix = "translate to english: "
            tibetan_inputs = [prefix + t for t in examples["tibetan"]]
            model_inputs = tokenizer(tibetan_inputs, max_length=max_input_length, truncation=True)

            ###########################
            # END add mt5 prefix
            ###########################

            ###########################
            # START use text_target rather than tokenizer target context
            ###########################

            # Original code
            # Set up the tokenizer for targets
            # with tokenizer.as_target_tokenizer():
            #     labels = tokenizer(examples["english"], max_length=max_input_length, truncation=True)

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            ###########################
            # END use text_target rather than tokenizer target context
            ###########################

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])
        print("Model readable dataset:", tokenized_dataset)

        return tokenized_dataset


# We add a custom Tibetan tokenizer in v2
class FinetuneMT5V2ExperimentBase(FinetuneMT5ExperimentBase):
    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained("buddhist-nlp/mt5-tibetan-tokenizer")

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        # Load pretrained parameter weights
        base_model_parameter_dict = AutoModelForSeq2SeqLM.from_pretrained(model_name).state_dict()
        base_model_parameter_dict = OrderedDict(base_model_parameter_dict)  # Make `base_model_parameter_dict` modifiable

        keys_to_modify = ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
        pretrained_embedding_weights = {k: base_model_parameter_dict.pop(k) for k in keys_to_modify}

        # Create new model
        config = AutoConfig.from_pretrained(model_name, vocab_size=tokenizer.vocab_size + 2)
        model = MT5Fp16ForConditionalGeneration(config)

        # Load pretrained weights into new model with a slight change to embeddings
        # since we have a larger vocab size
        model.load_state_dict(base_model_parameter_dict, strict=False)
        model_parameter_dict = model.state_dict()
        with torch.no_grad():
            for weight_name, pretrained_embedding_weight in pretrained_embedding_weights.items():
                pretrained_vocab_size, hidden_dim = pretrained_embedding_weight.size()
                model_parameter_dict[weight_name][:pretrained_vocab_size, :hidden_dim].copy_(pretrained_embedding_weight)

        # We don't have access to bf16 capable Ampere + GPUs so we need to workaround it
        scale_weights_for_fp16_t5(model)

        model.config.max_length = max_input_length

        return model


# We didn't use `model.prepare_decoder_input_ids` in our original implementation
# So we try using them now
class FinetuneMT5V3ExperimentBase(FinetuneMT5ExperimentBase):
    # This is an exact copy of `BaselineExperiment.run` unless specified otherwise
    def run(self, batch_size: int, learning_rates: List[float]) -> Dict[float, PredictionOutput]:
        max_input_length = self.MAX_INPUT_LENGTH
        trainer_cls = self.trainer_cls

        tokenizer = self.get_tokenizer()
        tokenized_dataset = self.load_data(tokenizer)
        compute_metrics = self.get_compute_metrics(tokenizer)

        ###############################
        # START remove previous model agnostic data collator creation
        ###############################

        # Original code
        data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
        
        ###############################
        # END remove previous model agnostic data collator creation
        ###############################

        # We perform hyperparam tuning over three learning rates
        for learning_rate in learning_rates:
            training_arguments = self.get_training_arguments(learning_rate, batch_size)

            print("Training with", training_arguments)
            model = self.get_model(tokenizer)

            ###############################
            # START add model-based data collator for use of `prepare_decoder_input_ids_from_labels`
            ###############################

            # No original code here

            data_collator = DataCollatorForSeq2Seq(
                tokenizer, model=model, max_length=max_input_length, padding="max_length"
            )

            ###############################
            # END add model-based data collator for use of `prepare_decoder_input_ids_from_labels`
            ###############################

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

            if training_arguments.local_rank <= 0:
                self._load_and_save_predictions_dict(learning_rate, predictions)

        return predictions


# We need to compare the results of fp32 vs our custom fp16 mt5 versions
class FinetuneMT5FP32ExperimentBase(FinetuneMT5ExperimentBase):
    # This is an exact copy of `FinetuneMT5ExperimentBase.get_model` unless specified otherwise
    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        ###############################
        # START don't use fp16 mt5
        ###############################

        # Original code
        # We don't have access to bf16 capable Ampere + GPUs so we need to workaround it
        # model = MT5Fp16ForConditionalGeneration.from_pretrained(model_name)
        # scale_weights_for_fp16_t5(model)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        ###############################
        # END don't use fp16 mt5
        ###############################

        model.config.max_length = max_input_length

        return model

    # This is an exact copy of `BaselinExperiment.get_training_arguments` unless specified otherwise
    def get_training_arguments(
        self, learning_rate: float, batch_size: int
    ) -> TrainingArguments:
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

            ###########################
            # START No fp16
            ###########################

            # Original code
            # fp16=True,
            fp16=False,

            ###########################
            # END No fp16
            ###########################

            log_level="error",
            logging_steps=1,
            predict_with_generate=True,
        )


class FinetuneMT5BaseExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"


class FinetuneMT5BaseV2Experiment(FinetuneMT5V2ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeV2Experiment(FinetuneMT5V2ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLV2Experiment(FinetuneMT5V2ExperimentBase):
    MODEL_NAME = "google/mt5-xl"


class FinetuneMT5BaseV3Experiment(FinetuneMT5V3ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeV3Experiment(FinetuneMT5V3ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLV3Experiment(FinetuneMT5V3ExperimentBase):
    MODEL_NAME = "google/mt5-xl"



class FinetuneMT5BaseFP32Experiment(FinetuneMT5FP32ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeFP32Experiment(FinetuneMT5FP32ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLFP32Experiment(FinetuneMT5FP32ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
