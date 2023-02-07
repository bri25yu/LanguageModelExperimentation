from typing import Callable, Dict, List, Sequence

from torch import randint

from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.training_pipelines import FinetuneExperimentBase
from lme.training_argument_mixins import MT5FinetuneArgsMixin
from lme.model_mixins import MT5Base580MModelMixin
from lme.experiments.translation.mixin import TranslationMixin


class TranslationIncomplete1Mixin(TranslationMixin):
    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

        def collate_fn(inputs: List[Dict[str, Sequence]]) -> Dict[str, List]:
            bools = randint(2, (len(inputs),))

            for to_truncate, d in zip(bools, inputs):
                if not to_truncate:
                    # 50% of the time, we leave the sequence as is
                    pass
                else:
                    # The other 50% of the time, we truncate the labels and append it to the input
                    # The amount of truncation here is uniformly random
                    truncation_amount = randint(len(d["label"]), ())
                    d["input_ids"] = d["input_ids"] + d["label"][:truncation_amount]

            return collator(inputs)

        return collate_fn


class TranslationIncompleteExperimentBase(MT5Base580MModelMixin, MT5FinetuneArgsMixin, FinetuneExperimentBase):
    pass


class TranslationIncomplete1Experiment(TranslationIncomplete1Mixin, TranslationIncompleteExperimentBase):
    pass
