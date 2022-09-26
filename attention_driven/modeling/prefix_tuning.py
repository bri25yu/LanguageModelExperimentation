import torch
from torch.nn.parameter import Parameter

from transformers import (
    M2M100ForConditionalGeneration, M2M100Config
)


class PrefixTuningM2M100Config(M2M100Config):
    def __init__(self, prefix_length: int=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prefix_length = prefix_length


class PrefixTuningNaiveM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    config_class = PrefixTuningM2M100Config

    def __init__(self, config) -> None:
        super().__init__(config)

        self.prefix_length = self.config.prefix_length

        # Freeze all our pretrained model parameters
        for param in self.parameters():
            param.requires_grad = False

        # Initialize prefix
        D = self.config.d_model
        std = self.config.init_std

        self.prefix = Parameter(torch.empty((self.prefix_length, D)))
        self.prefix.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # `input_ids` and `decoder_input_ids` are (batch_size, seq_len)
        if input_ids is not None:
            ids_to_embed = input_ids
            embed_model = self.get_encoder()
            batch_size, _ = input_ids.size()
        elif decoder_input_ids is not None:
            ids_to_embed = decoder_input_ids
            embed_model = self.get_decoder()
            batch_size, _ = decoder_input_ids.size()
        else:
            raise ValueError("Either input_ids or decoder_input_ids must not be None")

        # Convert our input_ids into input_embeds
        embeds = embed_model.embed_tokens(ids_to_embed) * embed_model.embed_scale

        # Prepend to our input_embeds
        batched_prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
        embeds_to_input = torch.concat((batched_prefix, embeds), dim=1)

        if input_ids is not None:
            # Set our input_ids to None
            input_ids = None

            # Set our inputs_embeds
            inputs_embeds = embeds_to_input

            prefix_length = self.prefix_length

            # Prepend to our attention mask
            prefix_attention_mask = torch.ones((batch_size, prefix_length), device=attention_mask.device)
            attention_mask = torch.concat((prefix_attention_mask, attention_mask), dim=1)

        elif decoder_input_ids is not None:
            # Set our input_ids to None
            decoder_input_ids = None

            # Set our inputs_embeds
            decoder_inputs_embeds = embeds_to_input

        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
