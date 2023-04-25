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

import math
import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Masked, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)


from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear


LIST = [
    "kssteven/ibert-roberta-base",
    "kssteven/ibert-roberta-large",
    "kssteven/ibert-roberta-large-mnli",
]


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class IBertEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.embedding_bit = 8
        self.embedding_act_bit = 16
        self.act_bit = 8
        self.ln_input_bit = 22
        self.ln_output_bit = 32

        self.word_embeddings = QuantEmbedding(
            config.s_vocab,
            config.d_model,
            padding_idx=config.PAD,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        self.token_type_embeddings = QuantEmbedding(
            config.n_typ,
            config.d_model,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.pos_type = getattr(config, "pos_type", "absolute")
        self.padding_idx = config.PAD
        self.position_embeddings = QuantEmbedding(
            config.n_pos,
            config.d_model,
            padding_idx=self.padding_idx,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.norm = IntLayerNorm(
            config.d_model,
            eps=config.eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.drop = qc.Dropout(config.drop)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds, inputs_embeds_scaling_factor = self.word_embeddings(input_ids)
        else:
            inputs_embeds_scaling_factor = None
        token_type_embeddings, token_type_embeddings_scaling_factor = self.token_type_embeddings(
            token_type_ids
        )

        embeddings, embeddings_scaling_factor = self.embeddings_act1(
            inputs_embeds,
            inputs_embeds_scaling_factor,
            identity=token_type_embeddings,
            identity_scaling_factor=token_type_embeddings_scaling_factor,
        )

        if self.pos_type == "absolute":
            position_embeddings, position_embeddings_scaling_factor = self.position_embeddings(
                position_ids
            )
            embeddings, embeddings_scaling_factor = self.embeddings_act1(
                embeddings,
                embeddings_scaling_factor,
                identity=position_embeddings,
                identity_scaling_factor=position_embeddings_scaling_factor,
            )

        embeddings, embeddings_scaling_factor = self.norm(embeddings, embeddings_scaling_factor)
        embeddings = self.drop(embeddings)
        embeddings, embeddings_scaling_factor = self.output_activation(
            embeddings, embeddings_scaling_factor
        )
        return embeddings, embeddings_scaling_factor

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class IBertSelfAttention(qc.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads != 0 and not hasattr(config, "d_embed"):
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_heads})"
            )
        self.quant_mode = config.quant_mode
        self.weight_bit = 8
        self.bias_bit = 32
        self.act_bit = 8

        self.n_heads = config.n_heads
        self.attention_head_size = int(config.d_model / config.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        # Q, K, V Linear layers
        self.query = QuantLinear(
            config.d_model,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.key = QuantLinear(
            config.d_model,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.value = QuantLinear(
            config.d_model,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )

        self.query_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.key_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.value_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        self.drop = qc.Dropout(config.drop_attn)
        self.pos_type = getattr(config, "pos_type", "absolute")
        if self.pos_type != "absolute":
            raise ValueError("I-BERT only supports 'absolute' for `config.pos_type`")

        self.softmax = IntSoftmax(
            self.act_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hiddens,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # Projection
        mixed_query_layer, mixed_query_layer_scaling_factor = self.query(
            hiddens, hidden_states_scaling_factor
        )
        mixed_key_layer, mixed_key_layer_scaling_factor = self.key(
            hiddens, hidden_states_scaling_factor
        )
        mixed_value_layer, mixed_value_layer_scaling_factor = self.value(
            hiddens, hidden_states_scaling_factor
        )

        # Requantization
        query_layer, query_layer_scaling_factor = self.query_activation(
            mixed_query_layer, mixed_query_layer_scaling_factor
        )
        key_layer, key_layer_scaling_factor = self.key_activation(
            mixed_key_layer, mixed_key_layer_scaling_factor
        )
        value_layer, value_layer_scaling_factor = self.value_activation(
            mixed_value_layer, mixed_value_layer_scaling_factor
        )

        # Transpose
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scale = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scale
        if self.quant_mode:
            attention_scores_scaling_factor = (
                query_layer_scaling_factor * key_layer_scaling_factor / scale
            )
        else:
            attention_scores_scaling_factor = None

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in IBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs, attention_probs_scaling_factor = self.softmax(
            attention_scores, attention_scores_scaling_factor
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.drop(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if attention_probs_scaling_factor is not None:
            context_layer_scaling_factor = (
                attention_probs_scaling_factor * value_layer_scaling_factor
            )
        else:
            context_layer_scaling_factor = None

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer, context_layer_scaling_factor = self.output_activation(
            context_layer, context_layer_scaling_factor
        )

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        output_scaling_factor = (
            (context_layer_scaling_factor, attention_probs_scaling_factor)
            if output_attentions
            else (context_layer_scaling_factor,)
        )

        return outputs, output_scaling_factor


class IBertSelfOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.ln_input_bit = 22
        self.ln_output_bit = 32

        self.dense = QuantLinear(
            config.d_model,
            config.d_model,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        self.norm = IntLayerNorm(
            config.d_model,
            eps=config.eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.drop = qc.Dropout(config.drop)

    def forward(
        self, hiddens, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor
    ):
        hiddens, hidden_states_scaling_factor = self.dense(hiddens, hidden_states_scaling_factor)
        hiddens = self.drop(hiddens)
        hiddens, hidden_states_scaling_factor = self.ln_input_act(
            hiddens,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        hiddens, hidden_states_scaling_factor = self.norm(hiddens, hidden_states_scaling_factor)

        hiddens, hidden_states_scaling_factor = self.output_activation(
            hiddens, hidden_states_scaling_factor
        )
        return hiddens, hidden_states_scaling_factor


class Attention(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.self = IBertSelfAttention(config)
        self.output = IBertSelfOutput(config)

    def forward(
        self,
        hiddens,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs, self_outputs_scaling_factor = self.self(
            hiddens,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output, attention_output_scaling_factor = self.output(
            self_outputs[0],
            self_outputs_scaling_factor[0],
            hiddens,
            hidden_states_scaling_factor,
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attns if we output them
        outputs_scaling_factor = (attention_output_scaling_factor,) + self_outputs_scaling_factor[
            1:
        ]
        return outputs, outputs_scaling_factor


class IBertIntermediate(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.dense = QuantLinear(
            config.d_model,
            config.d_ff,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        assert config.act == "gelu"
        self.intermediate_act_fn = IntGELU(
            quant_mode=self.quant_mode, force_dequant=config.force_dequant
        )
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(self, hiddens, hidden_states_scaling_factor):
        hiddens, hidden_states_scaling_factor = self.dense(hiddens, hidden_states_scaling_factor)
        hiddens, hidden_states_scaling_factor = self.intermediate_act_fn(
            hiddens, hidden_states_scaling_factor
        )

        hiddens, hidden_states_scaling_factor = self.output_activation(
            hiddens, hidden_states_scaling_factor
        )
        return hiddens, hidden_states_scaling_factor


class IBertOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.ln_input_bit = 22
        self.ln_output_bit = 32

        self.dense = QuantLinear(
            config.d_ff,
            config.d_model,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        self.norm = IntLayerNorm(
            config.d_model,
            eps=config.eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.drop = qc.Dropout(config.drop)

    def forward(
        self, hiddens, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor
    ):
        hiddens, hidden_states_scaling_factor = self.dense(hiddens, hidden_states_scaling_factor)
        hiddens = self.drop(hiddens)
        hiddens, hidden_states_scaling_factor = self.ln_input_act(
            hiddens,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        hiddens, hidden_states_scaling_factor = self.norm(hiddens, hidden_states_scaling_factor)

        hiddens, hidden_states_scaling_factor = self.output_activation(
            hiddens, hidden_states_scaling_factor
        )
        return hiddens, hidden_states_scaling_factor


class Layer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8

        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = IBertIntermediate(config)
        self.output = IBertOutput(config)

        self.pre_intermediate_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.pre_output_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(
        self,
        hiddens,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs, self_attention_outputs_scaling_factor = self.attention(
            hiddens,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]

        outputs = self_attention_outputs[1:]  # add self attns if we output attention weights

        layer_output, layer_output_scaling_factor = self.feed_forward_chunk(
            attention_output, attention_output_scaling_factor
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor):
        attention_output, attention_output_scaling_factor = self.pre_intermediate_act(
            attention_output, attention_output_scaling_factor
        )
        intermediate_output, intermediate_output_scaling_factor = self.intermediate(
            attention_output, attention_output_scaling_factor
        )

        intermediate_output, intermediate_output_scaling_factor = self.pre_output_act(
            intermediate_output, intermediate_output_scaling_factor
        )
        layer_output, layer_output_scaling_factor = self.output(
            intermediate_output,
            intermediate_output_scaling_factor,
            attention_output,
            attention_output_scaling_factor,
        )
        return layer_output, layer_output_scaling_factor


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quant_mode = config.quant_mode
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.n_lays)])

    def forward(
        self,
        hiddens,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None  # `config.add_cross_attention` is not supported
        next_decoder_cache = None  # `config.y_cache` is not supported

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hiddens,
                hidden_states_scaling_factor,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )

            hiddens = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hiddens,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return qo.CachesCrosses(
            y=hiddens,
            caches=next_decoder_cache,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
            crosses=all_cross_attentions,
        )


class Model(PreTrained):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.quant_mode = config.quant_mode
        self.embeddings = IBertEmbeddings(config)
        self.encoder = Encoder(config)
        self.pooler = Pool(config) if add_pooling_layer else None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [n_heads] or [n_lays x n_heads]
        # and head_mask is converted to shape [n_lays x batch x n_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)

        embedding_output, embedding_output_scaling_factor = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            embedding_output_scaling_factor,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return qo.BaseWithPoolingAndCrossAttentions(
            y=sequence_output,
            pools=pooled_output,
            caches=encoder_outputs.caches,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            crosses=encoder_outputs.crosses,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masked(**kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.ibert = Model(config)
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        flat_attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.ibert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.WithLoss(
            loss=loss,
            logits=reshaped_logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(cfg.d_model, "tanh", **kw)

    forward = qf.forward_seq


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
