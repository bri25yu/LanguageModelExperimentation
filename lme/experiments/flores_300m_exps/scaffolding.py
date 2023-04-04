from typing import Callable

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils.flores200 import get_flores_compute_metrics_cotr

from lme.data_processors.flores200 import (ScaffoldingOutputDataProcessor, ScaffoldingInputDataProcessor,
                                        ScaffoldingOutputMixDataProcessor, ScaffoldingInputMixDataProcessor,
                                        ScaffoldingInputMix2DataProcessor, ScaffoldingInputMix3DataProcessor,
                                        ScaffoldingOutputCOTRDataProcessor)


from lme.training_argument_mixins.utils import calculate_batch_size_args

from lme.experiments.flores_300m_exps.baseline import FloresBaselineMedium2Experiment


class FloresScaffoldExperimentBase(FloresBaselineMedium2Experiment):
    # DATA_PROCESSOR_CLS = 
    MAX_INPUT_LENGTH = 256

    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        args = super().get_training_arguments(batch_size=batch_size, learning_rate=learning_rate)

        args.max_steps = 10000

        target_total_batch_size_per_update = 2 ** 10 # 1024 batch size
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)

        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.per_device_train_batch_size = per_device_batch_size
        args.per_device_eval_batch_size = 2 * per_device_batch_size

        args.__post_init__()  # Reload hf deepspeed config

        return args


class FloresScaffoldInput300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingInputDataProcessor
    pass


class FloresScaffoldOutput300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingOutputDataProcessor
    pass


class FloresScaffoldInputMix300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingInputMixDataProcessor
    pass


class FloresScaffoldOutputMix300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingOutputMixDataProcessor
    pass


class FloresScaffoldInputMix2300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingInputMix2DataProcessor
    pass


class FloresScaffoldInputMix3300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingInputMix3DataProcessor
    pass


class FloresScaffoldOutputCOTR300MExperiment(FloresScaffoldExperimentBase):
    DATA_PROCESSOR_CLS = ScaffoldingOutputCOTRDataProcessor
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_flores_compute_metrics_cotr(tokenizer)

    pass
