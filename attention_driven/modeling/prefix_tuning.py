from types import MethodType

import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from transformers import (
    M2M100ForConditionalGeneration, M2M100Config
)
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.models.m2m_100.modeling_m2m_100 import (
    _expand_mask,
)


class PrefixTuningM2M100Config(M2M100Config):
    def __init__(self, prefix_length: int=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prefix_length = prefix_length


# This is an exact copy of the M2M100Encoder forward function except denoted otherwise
def prefix_tuning_encoder_forward(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.

            Indices can be obtained using [`M2M100Tokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

    embed_pos = self.embed_positions(input_ids, inputs_embeds)

    hidden_states = inputs_embeds + embed_pos

    ###################################
    # START prefix tuning hidden states
    ###################################

    # self.prefix is of size (L, D_p)
    # We need to expand it so that it matches the batch size of the hidden states
    # i.e. (L, D_p) -> (N, L, D_p)

    batch_size = input_ids.size()[0]
    batched_prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
    hidden_states = torch.concat((batched_prefix, hidden_states))

    ###################################
    # END prefix tuning hidden states
    ###################################

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is not None:

        ###################################
        # START prefix tuning attention mask
        ###################################

        prefix_attention_mask = torch.ones_like(batched_prefix, device=attention_mask.device)
        attention_mask = torch.concat((prefix_attention_mask, attention_mask))

        ###################################
        # END prefix tuning attention mask
        ###################################

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )


def create_prefix_parameter(L: int, D: int, std: float) -> nn.Module:
    prefix = Parameter(torch.empty((L, D)))

    prefix.data.normal_(mean=0.0, std=std)

    return prefix


class PrefixTuningM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    config_class = PrefixTuningM2M100Config

    def __init__(self, config) -> None:
        super().__init__(config)

        self.prefix_length = self.config.prefix_length

        # Freeze all our pretrained model parameters
        for param in self.parameters():
            param.requires_grad = False

        # Retrieve our encoder
        # The original prefix tuning paper prepends a prefix to both the
        # encoder and decoder, but we start with encoder only for now
        encoder = self.model.encoder

        # Add prefix params
        L = self.prefix_length
        D = self.config.d_model
        std = self.config.init_std

        encoder.prefix = create_prefix_parameter(L, D, std)

        # Modify the forward function of the encoder
        # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        encoder.forward = MethodType(prefix_tuning_encoder_forward, encoder)
