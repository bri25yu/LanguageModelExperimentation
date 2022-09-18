from typing import Optional, Tuple

from types import MethodType

import torch
import torch.nn as nn

from transformers import M2M100ForConditionalGeneration



# This is an exact copy of the M2M100Attention class except denoted otherwise
def lora_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj

    ###################################
    # START LoRA for query states
    ###################################

    # Original query states calculation
    # query_states = self.q_proj(hidden_states) * self.scaling

    lora_query_states = self.q_proj_b(self.q_proj_a(hidden_states))
    query_states = (self.q_proj(hidden_states) + lora_query_states) * self.scaling

    ###################################
    # END LoRA for query states
    ###################################

    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions

        ###################################
        # START LoRA for key states
        ###################################

        # Original key states calculation
        # key_states = self._shape(self.k_proj(key_value_states), -1, bsz)

        lora_key_states = self.k_proj_b(self.k_proj_a(key_value_states))
        key_states = self._shape(self.k_proj(key_value_states) + lora_key_states, -1, bsz)

        ###################################
        # END LoRA for key states
        ###################################

        ###################################
        # START LoRA for value states
        ###################################

        # Original key states calculation
        # value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        lora_value_states = self.v_proj_b(self.v_proj_a(key_value_states))
        value_states = self._shape(self.v_proj(key_value_states) + lora_value_states, -1, bsz)

        ###################################
        # END LoRA for value states
        ###################################

    elif past_key_value is not None:
        # reuse k, v, self_attention

        ###################################
        # START LoRA for key states
        ###################################

        # Original key states calculation
        # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)

        lora_key_states = self.k_proj_b(self.k_proj_a(hidden_states))
        key_states = self._shape(self.k_proj(hidden_states) + lora_key_states, -1, bsz)

        ###################################
        # END LoRA for key states
        ###################################

        ###################################
        # START LoRA for value states
        ###################################

        # Original key states calculation
        # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        lora_value_states = self.v_proj_b(self.v_proj_a(hidden_states))
        value_states = self._shape(self.v_proj(hidden_states) + lora_value_states, -1, bsz)

        ###################################
        # END LoRA for value states
        ###################################

        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        ###################################
        # START LoRA for key states
        ###################################

        # Original key states calculation
        # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)

        lora_key_states = self.k_proj_b(self.k_proj_a(hidden_states))
        key_states = self._shape(self.k_proj(hidden_states) + lora_key_states, -1, bsz)

        ###################################
        # END LoRA for key states
        ###################################

        ###################################
        # START LoRA for value states
        ###################################

        # Original key states calculation
        # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        lora_value_states = self.v_proj_b(self.v_proj_a(hidden_states))
        value_states = self._shape(self.v_proj(hidden_states) + lora_value_states, -1, bsz)

        ###################################
        # END LoRA for value states
        ###################################

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    ###################################
    # START LoRA for output states
    ###################################

    # Original output states calculation
    # attn_output = self.out_proj(attn_output)

    lora_output_states = self.out_proj_b(self.out_proj_a(hidden_states))
    attn_output = self.out_proj(attn_output) + lora_output_states

    ###################################
    # END LoRA for output states
    ###################################

    return attn_output, attn_weights_reshaped, past_key_value


class LoRAM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.rank = self.config.rank

        # Freeze all our pretrained model parameters
        for param in self.parameters():
            param.requires_grad = False

        # Next, we add our low-rank modules to our self attention layers
        for name, module in self.named_modules():
            if not (name.endswith("self_attn")):
                continue

            # Add our low-rank parameters
            self.add_lora_params(module)

            # Modify the forward function to actually use the low-rank parameters
            # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
            module.forward = MethodType(lora_forward, module)

    weight_prefixes = ["q", "k", "v", "out"]

    def add_lora_params(self, self_attn_module: nn.Module, rank: int) -> None:
        # See section 4.1 of the LoRA paper
        weight_prefixes = self.weight_prefixes
        rank = self.rank

        # Get our module shapes
        d = self_attn_module.embed_dim
        k = self_attn_module.embed_dim
        r = rank

        for weight_prefix in weight_prefixes:
            weight_name = f"{weight_prefix}_proj"

            # Create names for our A and B matrices
            b_name = f"{weight_name}_b"
            a_name = f"{weight_name}_a"

            # Create our A and B matrices
            B = nn.Linear(d, r)
            A = nn.Linear(r, k)

            # Initialize weights
            # Initialize all weights in B to be 0.
            # The bias is by default initialized to 0, so we don't touch it
            B.weight.data.zero_()
            # Apply Gaussian weight initialization
            self._init_weights(A)

            # Set A and B matrices as parameters of our self attn module
            setattr(self_attn_module, b_name, B)
            setattr(self_attn_module, a_name, A)
