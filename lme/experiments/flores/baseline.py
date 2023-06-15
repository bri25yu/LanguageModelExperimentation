from typing import Callable, List, Optional

import os

from datasets import DatasetDict, load_dataset

import evaluate

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from lme.training_argument_mixins.utils import (
    get_deepspeed_args, calculate_batch_size_args, get_default_training_arguments
)
from lme.model_mixins import MT5600MModelMixin, MT51BModelMixin, MT53BModelMixin
from lme.training_pipelines import FinetuneStagedTrainingArgsExperimentBase


class FloresStagedExperimentBase(FinetuneStagedTrainingArgsExperimentBase):
    MAX_INPUT_LENGTH = 256
    TRAINER_CLS = Seq2SeqTrainer
    DATASET_HF_PATHS: Optional[str] = None

    """
    This results in 10k * 2048 = 20mil examples.
    """
    TARGET_TOTAL_BATCH_SIZE_PER_UPDATE = 2 ** 11  # 2048

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding=True, pad_to_multiple_of=8)

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
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

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: Seq2SeqTrainingArguments) -> DatasetDict:
        pass

    def get_tokenized_datasets(self, tokenizer: PreTrainedTokenizerBase, training_arguments: Seq2SeqTrainingArguments) -> List[DatasetDict]:
        with training_arguments.main_process_first(desc="Loading data"):
            return list(map(lambda s: load_dataset(s, use_auth_token=True), self.DATASET_HF_PATHS))

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> Seq2SeqTrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )
        target_total_batch_size_per_update = self.TARGET_TOTAL_BATCH_SIZE_PER_UPDATE

        # !TODO: This is very poor engineering and programming practice. Sue me
        use_bf16 = os.environ.get("USE_BF16", "true")
        use_bf16 = use_bf16 == "true"

        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=200,
            save_steps=200,
            warmup_steps=0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=False,
            bf16=use_bf16,
            metric_for_best_model="chrf++",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )

    def update_training_arguments(self, training_arguments: Seq2SeqTrainingArguments, batch_size: int, stage: int) -> None:
        pass

    def update_data_collator(self, data_collator: Callable, stage: int) -> None:
        pass

    def update_model(self, model: PreTrainedModel, stage: int) -> None:
        pass


class FloresBaselineExperimentBase(FloresStagedExperimentBase):
    """
    DatasetDict({
        train: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 41412000
        })
        val: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['id', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 10000
        })
    })
    """
    DATASET_HF_PATHS = ["bri25yu/flores200_baseline_all_mt5"]


class FloresBaseline600MExperiment(MT5600MModelMixin, FloresBaselineExperimentBase):
    pass    


class FloresBaseline1BExperiment(MT51BModelMixin, FloresBaselineExperimentBase):
    pass


class FloresBaseline3BExperiment(MT53BModelMixin, FloresBaselineExperimentBase):
    pass
