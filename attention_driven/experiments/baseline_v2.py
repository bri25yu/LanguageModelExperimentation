from typing import Any, Callable, Dict

import evaluate

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizer

from attention_driven.experiments.baseline import BaselineExperiment
from attention_driven.data_processors import LDTibetanEnglishDataV2Processor
from attention_driven.data_processors.utils import convert_df_to_hf_dataset


class BaselineV2Experiment(BaselineExperiment):
    """
    This is the same as the baseline experiment with the following changes:
    - Add BLEU metrics
    - New silver data
    - Reduce the number of passes through the data. We moved from a train set size of 70k to 200k, so we decrease the number of epochs from 25 to 10.

    This experiment finetunes an NLLB 600M model
    """

    NUM_TRAIN_EPOCHS = 10

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

    # This is the exact same function as BaselineExperiment unless noted otherwise
    def load_data(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        This function assumes that https://github.com/Linguae-Dharmae/language-models
        has been cloned into the same root folder.
        """
        val_split_size = self.VAL_SPLIT_SIZE
        max_input_length = self.MAX_INPUT_LENGTH

        # Load our datasets from disk into HF Dataset's

        ###############################
        # START change the data processor
        ###############################

        # Original code
        # data_processor = LDTibetanEnglishDataProcessor()

        data_processor = LDTibetanEnglishDataV2Processor()

        ###############################
        # END change the data processor
        ###############################

        train_dataset, test_dataset = convert_df_to_hf_dataset(data_processor())
        train_val_dataset = train_dataset.train_test_split(val_split_size, seed=42)

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
