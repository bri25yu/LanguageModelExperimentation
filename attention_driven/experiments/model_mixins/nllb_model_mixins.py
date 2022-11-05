from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class NLLBModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # We use a unidirectional tokenizer from Tibetan to English
        tokenizer.src_lang = "bod_Tibt"
        tokenizer.tgt_lang = "eng_Latn"

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        assert model_name, f"Must override `MODEL_NAME` attribute of {self.name}"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        model._keys_to_ignore_on_save = []

        return model


class NLLB600MModelMixin(NLLBModelMixinBase):
    MODEL_NAME = "facebook/nllb-200-distilled-600M"


class NLLB1_3BModelMixin(NLLBModelMixinBase):
    MODEL_NAME = "facebook/nllb-200-1.3B"
