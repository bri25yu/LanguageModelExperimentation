from typing import Any, Dict

import os

import json

from transformers import TrainingArguments, Seq2SeqTrainingArguments

from lme import CONFIG_DIR


__all__ = ["get_seq2seq_training_arguments", "calculate_total_examples"]


def get_world_size() -> int:
    training_arguments = TrainingArguments("")
    return training_arguments.world_size


def get_deepspeed_args(scheduler_type: str) -> Dict[str, Any]:
    try:
        import deepspeed

        has_deepspeed = True
    except ImportError:
        has_deepspeed = False

    if has_deepspeed:
        deepspeed_args_path = os.path.join(CONFIG_DIR, "deepspeed.json")
    else:
        deepspeed_args_path = None

    deepspeed_args = json.load(open(deepspeed_args_path))

    # Logic to work around the deepspeed scheduler config
    ALLOWED_SCHEDULERS = ["WarmupLR", "WarmupDecayLR"]
    assert scheduler_type in ALLOWED_SCHEDULERS, f"Unrecognize Deepspeed LR scheduler {scheduler_type}. Allowed scheduler types include {ALLOWED_SCHEDULERS}"
    deepspeed_args["scheduler"]["type"] = scheduler_type
    if scheduler_type == "WarmupLR":
        if "total_num_steps" in deepspeed_args["scheduler"]["params"]:
            del deepspeed_args["scheduler"]["params"]["total_num_steps"]

    return deepspeed_args


def get_seq2seq_training_arguments(
    base_output_dir: str,
    learning_rate: float,
    max_steps: int,
    eval_steps: int,
    target_total_batch_size_per_update: int,
    per_gpu_batch_size: int,
    scheduler_type: str,
    warmup_steps: int,
) -> Seq2SeqTrainingArguments:
    """
    Standardized Seq2Seq training arguments constructor
    """
    output_dir = os.path.join(
        base_output_dir, f"{learning_rate:.0e}"
    )
    eval_save_strategy = "steps"

    world_size = get_world_size()
    gradient_accumulation_steps = target_total_batch_size_per_update // (per_gpu_batch_size * world_size)
    if gradient_accumulation_steps < 1:
        # We ensure our true target batch size is constant, irregardless of hardware restrictions
        gradient_accumulation_steps = 1
        per_gpu_batch_size = target_total_batch_size_per_update // world_size

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
        warmup_steps=warmup_steps,
        deepspeed=get_deepspeed_args(scheduler_type),
    )


def calculate_total_examples(args: TrainingArguments) -> int:
    world_size = get_world_size()
    return args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * world_size
