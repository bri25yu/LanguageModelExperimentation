from typing import Any, Dict, Tuple

import os

import json

from transformers import TrainingArguments

from lme import CONFIG_DIR


__all__ = [
    "get_deepspeed_args",
    "calculate_total_examples",
    "calculate_batch_size_args",
    "get_default_training_arguments",
]


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


def calculate_batch_size_args(target_total_batch_size_per_update: int, per_device_batch_size: int) -> Tuple[int, int]:
    world_size = get_world_size()
    gradient_accumulation_steps = target_total_batch_size_per_update // (per_device_batch_size * world_size)
    if gradient_accumulation_steps < 1:
        # We ensure our true target batch size is constant, irregardless of hardware restrictions
        gradient_accumulation_steps = 1
        per_device_batch_size = target_total_batch_size_per_update // world_size

    return gradient_accumulation_steps, per_device_batch_size


def get_default_training_arguments() -> Dict[str, Any]:
    training_arguments_path = os.path.join(CONFIG_DIR, "training_arguments.json")

    return json.load(open(training_arguments_path))


def calculate_total_examples(args: TrainingArguments) -> int:
    world_size = get_world_size()
    return args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * world_size
