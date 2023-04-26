# Copyright 2022 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import copy
import math
import random
import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained


from torch.nn import CrossEntropyLoss

log = logging.get_logger(__name__)


LIST = [
    "uclanlp/plbart-base",
    "uclanlp/plbart-cs-java",
    "uclanlp/plbart-multi_task-all",
]


# Copied from transformers.models.mbart.modeling_mbart.shift_tokens_right
def shift_tokens_right(input_ids, PAD):
    prev_output_tokens = input_ids.clone()

    if PAD is None:
        raise ValueError("self.model.config.PAD has to be defined.")
    # replace possible -100 values in labels by `PAD`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, PAD)

    index_of_eos = (prev_output_tokens.ne(PAD).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


# Copied from transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding with Bart->PLBart
class PLBartLearnedPositionalEmbedding(qc.Embed):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape, past_key_values_length=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PLBart
class Attention(qc.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        drop: float = 0.0,
        is_decoder=False,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.drop = drop
        self.head_dim = embed_dim // n_heads

        if (self.head_dim * n_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by n_heads (got `embed_dim`: {self.embed_dim}"
                f" and `n_heads`: {n_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = qc.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = qc.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = qc.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = qc.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hiddens,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hiddens.size()

        # get query proj
        query_states = self.q_proj(hiddens) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, crosses
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # crosses
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hiddens), -1, bsz)
            value_states = self._shape(self.v_proj(hiddens), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hiddens), -1, bsz)
            value_states = self._shape(self.v_proj(hiddens), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.n_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.n_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.n_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.n_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.n_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.n_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.n_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.n_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.n_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.n_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.n_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.drop(attn_weights, p=self.drop, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.n_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.n_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.n_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.bart.modeling_bart.BartEncoderLayer with Bart->PLBart
class EncLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            n_heads=config.encoder_attention_heads,
            drop=config.drop_attn,
        )
        self.self_attn_layer_norm = qc.LayerNorm(self.embed_dim)
        self.drop = config.drop
        self.activation_fn = qu.activation(config.act)
        self.drop_act = config.drop_act
        self.fc1 = qc.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = qc.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = qc.LayerNorm(self.embed_dim)

    def forward(
        self,
        hiddens,
        attention_mask,
        layer_head_mask,
        output_attentions=False,
    ):
        residual = hiddens
        hiddens, attn_weights, _ = self.self_attn(
            hiddens=hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        hiddens = residual + hiddens
        hiddens = self.self_attn_layer_norm(hiddens)

        residual = hiddens
        hiddens = self.activation_fn(self.fc1(hiddens))
        hiddens = F.drop(hiddens, p=self.drop_act, training=self.training)
        hiddens = self.fc2(hiddens)
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        hiddens = residual + hiddens
        hiddens = self.final_layer_norm(hiddens)

        if hiddens.dtype == torch.float16 and (
            torch.isinf(hiddens).any() or torch.isnan(hiddens).any()
        ):
            clamp_value = torch.finfo(hiddens.dtype).max - 1000
            hiddens = torch.clamp(hiddens, min=-clamp_value, max=clamp_value)

        outputs = (hiddens,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->PLBart
class DecLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            n_heads=config.decoder_attention_heads,
            drop=config.drop_attn,
            is_decoder=True,
        )
        self.drop = config.drop
        self.activation_fn = qu.activation(config.act)
        self.drop_act = config.drop_act

        self.self_attn_layer_norm = qc.LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            drop=config.drop_attn,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = qc.LayerNorm(self.embed_dim)
        self.fc1 = qc.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = qc.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = qc.LayerNorm(self.embed_dim)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        y_cache=True,
    ):
        residual = hiddens
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hiddens, self_attn_weights, present_key_value = self.self_attn(
            hiddens=hiddens,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        hiddens = residual + hiddens
        hiddens = self.self_attn_layer_norm(hiddens)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if enc_hiddens is not None:
            residual = hiddens

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hiddens, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hiddens=hiddens,
                key_value_states=enc_hiddens,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hiddens = F.drop(hiddens, p=self.drop, training=self.training)
            hiddens = residual + hiddens
            hiddens = self.encoder_attn_layer_norm(hiddens)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hiddens
        hiddens = self.activation_fn(self.fc1(hiddens))
        hiddens = F.drop(hiddens, p=self.drop_act, training=self.training)
        hiddens = self.fc2(hiddens)
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        hiddens = residual + hiddens
        hiddens = self.final_layer_norm(hiddens)

        outputs = (hiddens,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if y_cache:
            outputs += (present_key_value,)

        return outputs


class Encoder(PreTrained):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.drop = config.drop
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.PAD
        self.max_source_positions = config.n_pos
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = qc.Embed(config.s_vocab, embed_dim, self.padding_idx)

        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.n_pos,
            embed_dim,
        )
        self.layers = nn.ModuleList([EncLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = qc.LayerNorm(embed_dim)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
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

        embed_pos = self.embed_positions(input_shape)

        hiddens = inputs_embeds + embed_pos
        hiddens = self.layernorm_embedding(hiddens)
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            attention_mask = qu.expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hiddens,)
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
                        hiddens,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hiddens,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hiddens = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hiddens,)

        if not return_dict:
            return tuple(v for v in [hiddens, encoder_states, all_attentions] if v is not None)
        return qo.Base(y=hiddens, hiddens=encoder_states, attns=all_attentions)


# Copied from transformers.models.bart.modeling_bart.BartDecoder with Bart->PLBart
class Decoder(PreTrained):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.drop = config.drop
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.PAD
        self.max_target_positions = config.n_pos
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = qc.Embed(config.s_vocab, config.d_model, self.padding_idx)

        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.n_pos,
            config.d_model,
        )
        self.layers = nn.ModuleList([DecLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = qc.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = qu.causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            expanded_attn_mask = qu.expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        caches=None,
        inputs_embeds=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        y_cache = y_cache if y_cache is not None else self.config.y_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = caches[0][0].shape[2] if caches is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if enc_hiddens is not None and encoder_attention_mask is not None:
            encoder_attention_mask = qu.expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hiddens = inputs_embeds + positions
        hiddens = self.layernorm_embedding(hiddens)

        hiddens = F.drop(hiddens, p=self.drop, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and enc_hiddens is not None) else None
        next_decoder_cache = () if y_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hiddens,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = caches[idx] if caches is not None else None

            if self.gradient_checkpointing and self.training:
                if y_cache:
                    log.warning(
                        "`y_cache=True` is incompatible with gradient checkpointing. Setting `y_cache=False`..."
                    )
                    y_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, y_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hiddens,
                    attention_mask,
                    enc_hiddens,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hiddens,
                    attention_mask=attention_mask,
                    enc_hiddens=enc_hiddens,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    y_cache=y_cache,
                )
            hiddens = layer_outputs[0]

            if y_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if enc_hiddens is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hiddens,)

        next_cache = next_decoder_cache if y_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hiddens,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return qo.CachesCrosses(
            y=hiddens,
            caches=next_cache,
            hiddens=all_hidden_states,
            attns=all_self_attns,
            crosses=all_cross_attentions,
        )


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        padding_idx, s_vocab = config.PAD, config.s_vocab
        self.shared = qc.Embed(s_vocab, config.d_model, padding_idx)

        self.encoder = Encoder(config, self.shared)
        self.decoder = Decoder(config, self.shared)

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
        caches=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        y_cache = y_cache if y_cache is not None else self.config.y_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, PLBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.PAD)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                y=encoder_outputs[0],
                hiddens=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attns=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            enc_hiddens=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            caches=caches,
            inputs_embeds=decoder_inputs_embeds,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            y=decoder_outputs.y,
            caches=decoder_outputs.caches,
            hiddens=decoder_outputs.hiddens,
            attns=decoder_outputs.attns,
            crosses=decoder_outputs.crosses,
            enc_y=encoder_outputs.y,
            enc_hiddens=encoder_outputs.hiddens,
            enc_attns=encoder_outputs.attns,
        )


class ForCondGen(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = qc.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

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
        caches=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.PAD)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            caches=caches,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.s_vocab), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            caches=outputs.caches,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
            enc_y=outputs.enc_y,
            enc_hiddens=outputs.enc_hiddens,
            enc_attns=outputs.enc_attns,
        )


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, "tanh", **kw)

    forward = qf.forward_seq

    def pre_proj(self, x, ys):
        y = ys[0]
        eos_m = x.eq(self.cfg.EOS)
        assert len(torch.unique_consecutive(eos_m.sum(1))) <= 1
        y = y[eos_m, :].view(y.size(0), -1, y.size(-1))
        return y[:, -1, :]


# Copied from transformers.models.bart.modeling_bart.BartDecoderWrapper with Bart->PLBart
class PLBartDecoderWrapper(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = Decoder(config)

    def forward(self, *args, **kw):
        return self.decoder(*args, **kw)


# Copied from transformers.models.bart.modeling_bart.ForCausal with Bart->PLBart, facebook/bart-base->uclanlp/plbart-base
class ForCausal(PreTrained):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_enc_dec = False
        super().__init__(config)
        self.model = PLBartDecoderWrapper(config)
        self.lm_head = qc.Linear(config.d_model, config.s_vocab, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        caches=None,
        inputs_embeds=None,
        labels=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_hiddens=enc_hiddens,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            caches=caches,
            inputs_embeds=inputs_embeds,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.s_vocab), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            caches=outputs.caches,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )
