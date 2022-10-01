from transformers import Trainer
from transformers.deepspeed import deepspeed_init


__all__ = ["init_deepspeed_inference"]


def init_deepspeed_inference(trainer: Trainer) -> None:
    """
    Currently, running inference with deepspeed without first training requires zero 3. This is a workaround
    """
    deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
        trainer, num_training_steps=0
    )
    trainer.model = deepspeed_engine.module
    trainer.model_wrapped = deepspeed_engine
    trainer.deepspeed = deepspeed_engine
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
