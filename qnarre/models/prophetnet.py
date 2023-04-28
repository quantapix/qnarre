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
import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embed
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained

from dataclasses import dataclass

from ...modeling_utils import PreTrained

log = logging.get_logger(__name__)

LIST = [
    "microsoft/prophetnet-large-uncased",
]


def softmax(hidden_state, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(hidden_state.float(), dim=dim)
    else:
        return F.softmax(hidden_state, dim=dim, dtype=torch.float32)


def ngram_attention_bias(sequence_length, ngram, device, dtype):
    left_block = torch.ones(
        (ngram, sequence_length, sequence_length), device=device, dtype=dtype
    ) * float("-inf")
    right_block = left_block.detach().clone()
    # create bias
    for stream_idx in range(ngram):
        right_block[stream_idx].fill_diagonal_(0, wrap=False)
        left_block[stream_idx].triu_(-stream_idx + 1)

    left_block[:, :, 0] = 0
    return torch.cat([left_block, right_block], dim=2)


def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = (
            rel_positions_bucket
            + torch.lt(inv_relative_positions, torch.zeros_like(inv_relative_positions)).int()
            * num_buckets
        )
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        inv_relative_positions = torch.max(
            inv_relative_positions, torch.zeros_like(inv_relative_positions)
        )

    max_exact = num_buckets // 2
    is_small = torch.lt(inv_relative_positions, max_exact)
    val_if_large = max_exact + torch.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + torch.where(
        is_small, inv_relative_positions.int(), val_if_large
    )
    return rel_positions_bucket


def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    # main stream
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # predicting stream
    predicting_stream_relative_positions = torch.cat(
        (position_ids - 1, position_ids), dim=-1
    ).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(
        1, position_ids.size(-1), 1
    )
    predicting_stream_relative_positions = (
        predicting_stream_relative_positions - position_ids.unsqueeze(-1)
    )

    # get both position buckets
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    return main_relative_position_buckets, predict_relative_position_buckets


@dataclass
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    loss = None
    logits = None
    logits_ngram = None
    caches = None
    hiddens = None
    decoder_ngram_hidden_states = None
    attns = None
    decoder_ngram_attentions = None
    crosses = None
    enc_y = None
    enc_hiddens = None
    enc_attns = None


@dataclass
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    y
    last_hidden_state_ngram = None
    caches = None
    hiddens = None
    decoder_ngram_hidden_states = None
    attns = None
    decoder_ngram_attentions = None
    crosses = None
    enc_y = None
    enc_hiddens = None
    enc_attns = None


@dataclass
class ProphetNetDecoderModelOutput(ModelOutput):
    y
    last_hidden_state_ngram = None
    caches = None
    hiddens = None
    hidden_states_ngram = None
    attns = None
    ngram_attentions = None
    crosses = None


@dataclass
class ProphetNetDecoderLMOutput(ModelOutput):
    loss = None
    logits = None
    logits_ngram = None
    caches = None
    hiddens = None
    hidden_states_ngram = None
    attns = None
    ngram_attentions = None
    crosses = None


class ProphetNetPositionalEmbeddings(qc.Embed):
    def __init__(self, config):
        self.max_length = config.n_pos
        super().__init__(config.n_pos, config.d_model, config.PAD)

    def forward(self, inputs_shape, device, attention_mask=None, caches=None, position_ids=None):
        assert (position_ids is None) or (self.padding_idx is None)
        if position_ids is None:
            if caches is not None:
                prev_num_input_ids = caches[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx
                position_ids = position_ids.clamp(0, self.max_length - 1)

        return super().forward(position_ids), position_ids

    def _forward(self, position_ids):
        return super().forward(position_ids)


class Attention(qc.Module):
    def __init__(
        self,
        config,
        num_attn_heads,
    ):
        super().__init__()
        d_model = config.d_model

        self.drop_attn = config.drop_attn
        self.drop = config.drop
        self.num_attn_heads = num_attn_heads
        self.head_dim = d_model // num_attn_heads

        assert self.head_dim * num_attn_heads == d_model

        self.key_proj = qc.Linear(d_model, d_model)
        self.value_proj = qc.Linear(d_model, d_model)
        self.query_proj = qc.Linear(d_model, d_model)

        self.out_proj = qc.Linear(d_model, d_model)

    def _shape(self, tensor, seq_len, bsz):
        return (
            tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hiddens,
        key_value_states=None,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        batch_size, tgt_len, d_model = hiddens.size()
        is_cross_attention = key_value_states is not None
        assert list(hiddens.size()) == [
            batch_size,
            tgt_len,
            d_model,
        ]
        query_states = self.query_proj(hiddens) / (self.head_dim**0.5)

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, crosses
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # crosses
            key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.key_proj(hiddens), -1, batch_size)
            value_states = self._shape(self.value_proj(hiddens), -1, batch_size)

        if is_cross_attention:
            past_key_value = (key_states, value_states)
        proj_shape = (batch_size * self.num_attn_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (
            batch_size * self.num_attn_heads,
            tgt_len,
            src_len,
        )
        if attention_mask is not None and attention_mask.dim() == 0:
            attention_mask = None
        assert attention_mask is None or attention_mask.size() == (
            self.num_attn_heads * batch_size,
            1,
            src_len,
        )

        if attention_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights + attention_mask

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_attn_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_attn_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,)
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                batch_size, self.num_attn_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(batch_size * self.num_attn_heads, tgt_len, src_len)
            attn_weights_reshaped = layer_head_mask.view(1, -1, 1, 1) * attn_weights_reshaped

        attn_probs = F.drop(
            attn_weights,
            p=self.drop_attn,
            training=self.training,
        )

        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (
            batch_size * self.num_attn_heads,
            tgt_len,
            self.head_dim,
        )

        attn_output = (
            attn_output.view(batch_size, self.num_attn_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, tgt_len, d_model)
        )

        attn_output = self.out_proj(attn_output)

        attn_output = F.drop(attn_output, p=self.drop, training=self.training)
        return attn_output, attn_weights_reshaped, past_key_value


class ProphetNetFeedForward(qc.Module):
    def __init__(self, config, ffn_dim):
        super().__init__()
        self.act = qu.activation(config.act)
        self.intermediate = qc.Linear(config.d_model, ffn_dim)
        self.output = qc.Linear(ffn_dim, config.d_model)
        self.drop_act = config.drop_act
        self.drop = config.drop

    def forward(self, x):
        y = self.intermediate(x)
        y = self.act(y)
        y = F.drop(y, p=self.drop_act, training=self.training)
        y = self.output(y)
        y = F.drop(y, p=self.drop, training=self.training)
        return y


class ProphetNetNgramSelfAttention(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.num_attn_heads = config.num_decoder_attention_heads
        self.drop = config.drop
        self.drop_attn = config.drop_attn
        self.head_dim = config.d_model // self.num_attn_heads
        self.ngram = config.ngram

        assert self.head_dim * self.num_attn_heads == config.d_model
        self.key_proj = qc.Linear(config.d_model, config.d_model)
        self.value_proj = qc.Linear(config.d_model, config.d_model)
        self.query_proj = qc.Linear(config.d_model, config.d_model)
        self.out_proj = qc.Linear(config.d_model, config.d_model)
        self.relative_pos_embeddings = qc.Linear(
            config.d_model, self.num_buckets * self.num_attn_heads
        )
        self.onnx_trace = False

    def _shape(self, tensor, seq_len, batch_size):
        return (
            tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hiddens,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
        batch_size, ngram_sequence_length, d_model = hiddens.size()

        assert list(hiddens.size()) == [
            batch_size,
            ngram_sequence_length,
            d_model,
        ]

        # project
        query_states = self.query_proj(hiddens)
        key_states = self.key_proj(hiddens)
        value_states = self.value_proj(hiddens)

        # normalize
        query_states = query_states / (self.head_dim**0.5)

        # reshape
        query_states = self._shape(query_states, ngram_sequence_length, batch_size)
        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_attn_heads, -1, self.head_dim)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # chunk into main stream and predict stream
        hidden_states_list = hiddens.chunk(1 + self.ngram, dim=1)

        query_states_list = query_states.chunk(1 + self.ngram, dim=1)
        key_states_list = key_states.chunk(1 + self.ngram, dim=1)
        value_states_list = value_states.chunk(1 + self.ngram, dim=1)

        main_hidden_states, hidden_states_predict_list = (
            hidden_states_list[0],
            hidden_states_list[1:],
        )
        main_query_states, predict_query_states_list = query_states_list[0], query_states_list[1:]
        main_key_states, predict_key_states_list = key_states_list[0], key_states_list[1:]
        main_value_states, predict_value_states_list = value_states_list[0], value_states_list[1:]

        # saved states are stored with shape (batch_size, num_attn_heads, seq_len, head_dim)
        if past_key_value is not None:
            prev_main_key_states = past_key_value[0].view(
                batch_size * self.num_attn_heads, -1, self.head_dim
            )
            main_key_states = torch.cat((prev_main_key_states, main_key_states), dim=1)
            prev_main_value_states = past_key_value[1].view(
                batch_size * self.num_attn_heads, -1, self.head_dim
            )
            main_value_states = torch.cat((prev_main_value_states, main_value_states), dim=1)

        # Update cache
        past_key_value = (
            main_key_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
            main_value_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
        )

        # get seq_length of main stream only
        sequence_length = ngram_sequence_length // (1 + self.ngram)

        # MAIN-STREAM
        # main attn weights
        main_attn_weights = torch.bmm(main_query_states, main_key_states.transpose(1, 2))

        main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(
            main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets
        )
        main_attn_weights = main_attn_weights + main_relative_pos_embeddings

        if attention_mask is not None:
            main_attn_weights = main_attn_weights + attention_mask

        main_attn_probs = softmax(
            main_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(main_attn_weights)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,)
            main_attn_probs = layer_head_mask.view(1, -1, 1, 1) * main_attn_probs.view(
                batch_size, self.num_attn_heads, -1, sequence_length
            )
            main_attn_probs = main_attn_probs.view(
                batch_size * self.num_attn_heads, -1, sequence_length
            )

        main_attn_probs = F.drop(main_attn_probs, p=self.drop_attn, training=self.training)
        # project to attn_output
        main_attn_output = torch.bmm(main_attn_probs, main_value_states)

        # reshape so that n_heads dim is merged into last `head_dim` axis
        main_attn_output = (
            main_attn_output.view(batch_size, self.num_attn_heads, sequence_length, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, 1, sequence_length, d_model)
        )
        main_attn_output = self.out_proj(main_attn_output)

        # PREDICT-STREAM
        # [ngram, B*head, T, c]
        predict_query_states = torch.cat(predict_query_states_list, 0).view(
            self.ngram, -1, sequence_length, self.head_dim
        )
        # [ngram, B*head, 2*T, c]
        predict_key_states = torch.cat(
            [torch.cat([main_key_states, key], 1).unsqueeze(0) for key in predict_key_states_list],
            0,
        )

        # [ngram, T, B, C]
        predict_hidden_states = torch.cat(hidden_states_predict_list, 0).view(
            self.ngram, sequence_length, batch_size, d_model
        )

        # [ngram, B*head, 2*T, c]
        predict_value_states = torch.cat(
            [
                torch.cat([main_value_states, v_p], 1).unsqueeze(0)
                for v_p in predict_value_states_list
            ],
            0,
        )
        # [ngram, B*head, T, 2*T]
        predict_attn_weights = torch.einsum(
            "nbtc,nbsc->nbts", (predict_query_states, predict_key_states)
        )

        predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
            predict_hidden_states,
            predict_attn_weights,
            position_ids,
            predict_relative_position_buckets,
        )

        # [ngram, B*head, T, 2*T]
        predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings

        if extended_predict_attention_mask is not None:
            predict_attn_weights = predict_attn_weights + extended_predict_attention_mask.to(
                predict_attn_weights.dtype
            )

        predict_attn_probs = softmax(
            predict_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(predict_attn_weights)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,)
            predict_attn_probs = layer_head_mask.view(1, 1, -1, 1, 1) * predict_attn_probs.view(
                self.ngram, batch_size, self.num_attn_heads, sequence_length, 2 * sequence_length
            )
            predict_attn_probs = predict_attn_probs.view(
                self.ngram, batch_size * self.num_attn_heads, sequence_length, 2 * sequence_length
            )

        predict_attn_probs = F.drop(predict_attn_probs, p=self.drop_attn, training=self.training)
        # project to attention output
        # [ngram, B*head, T, c]
        predict_attn_output = torch.einsum(
            "nbts,nbsc->nbtc", (predict_attn_probs, predict_value_states)
        )

        # reshape so that n_heads dim is merged into last `head_dim` axis
        # [ngram, B, T, C]
        predict_attn_output = (
            predict_attn_output.view(
                self.ngram, batch_size, self.num_attn_heads, sequence_length, self.head_dim
            )
            .permute(1, 0, 3, 2, 4)
            .reshape(batch_size, self.ngram, sequence_length, d_model)
        )
        predict_attn_output = self.out_proj(predict_attn_output)

        # concat to single attn output
        # [B, 1+ngram*T, C]
        attn_output = torch.cat([main_attn_output, predict_attn_output], 1).view(
            batch_size, -1, d_model
        )
        # reshape into better form for `config.output_attentions`
        main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, sequence_length, -1)
        predict_attn_probs = predict_attn_probs.view(
            self.ngram, batch_size, self.num_attn_heads, sequence_length, -1
        ).transpose(0, 1)

        attn_output = F.drop(attn_output, p=self.drop, training=self.training)

        return attn_output, main_attn_probs, predict_attn_probs, past_key_value

    def get_main_relative_pos_embeddings(
        self, hiddens, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hiddens [B,T,C], input attn_weights [T*head,T,S], input position_ids [B,T] or [1,1]

        if main_relative_position_buckets is None:
            batch_size, sequence_length = hiddens.shape[:2]
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(
                batch_size, sequence_length, 1
            )  # [B, T, s]
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        rel_pos_embeddings = self.relative_pos_embeddings(hiddens)  # [B,T,Buckets*head]
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        ).permute(
            0, 3, 1, 2
        )  # [B,T,Buckets,head]
        rel_pos_embeddings = rel_pos_embeddings.reshape(
            attn_weights.shape[:2] + (-1,)
        )  # [B*head,T,Buckets]

        main_relative_position_buckets = (
            main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
            .view(-1, main_relative_position_buckets.shape[-1])
            .long()
        )  # [B*head*T, T]
        rel_pos_embeddings = rel_pos_embeddings.reshape(
            -1, rel_pos_embeddings.size(-1)
        )  # [B*head*T,Buckets]

        main_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=main_relative_position_buckets
        ).view(attn_weights.shape[:2] + (-1,))

        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hiddens, attn_weights, position_ids, predict_relative_position_buckets
    ):
        # input hiddens [ngram, T,B,C], input attn_weights [ngram, B*head,T,S], input position_ids [B,T] or [1,1], input predict_relative_position_buckets [B,T, 2*T] or None
        sequence_length, batch_size = hiddens.shape[1:3]

        if predict_relative_position_buckets is None:
            key_sequence_length = attn_weights.shape[-1]
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            relative_positions = (
                torch.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )

            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(
                batch_size, sequence_length, 1
            )
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        hiddens = hiddens.transpose(1, 2)  # [ngram, B, T, C]
        rel_pos_embeddings = self.relative_pos_embeddings(hiddens).view(
            hiddens.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )  # [ngram, B, T, bucket, head]
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 1, 4, 2, 3).reshape(
            self.ngram * batch_size * self.num_attn_heads, sequence_length, -1
        )  # [ngram*B*head, T, bucket]

        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0).repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )  # [ngram, B, head*T, S]

        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.size(-1)
        ).long()  # [ngram*B*head*T, S]

        predict_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
        ).view(
            self.ngram, batch_size * self.num_attn_heads, sequence_length, -1
        )  # [ngram, B*head, T, S]

        return predict_relative_pos_embeddings


class EncLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        # 1st residual block
        self.self_attn = Attention(config, config.num_encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(config.d_model)

        # 2nd residual block
        self.feed_forward = ProphetNetFeedForward(config, config.encoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.d_model)

    def forward(
        self,
        hiddens,
        attention_mask,
        layer_head_mask,
        output_attentions=False,
    ):
        # 1st residual block
        attention_output, attn_weights, _ = self.self_attn(
            hiddens=hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hiddens = self.self_attn_layer_norm(attention_output + hiddens)

        # 2nd residual block
        feed_forward_output = self.feed_forward(hiddens)
        hiddens = self.feed_forward_layer_norm(feed_forward_output + hiddens)

        outputs = (hiddens,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DecLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        # 1st residual block
        self.self_attn = ProphetNetNgramSelfAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.d_model)

        # 2nd residual block
        if config.add_cross_attention:
            self.cross_attn = Attention(config, config.num_decoder_attention_heads)
            self.cross_attn_layer_norm = LayerNorm(config.d_model)

        # 3rd residual block
        self.feed_forward = ProphetNetFeedForward(config, config.decoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.d_model)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        enc_hiddens=None,
        encoder_attn_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
        past_key_value=None,
        y_cache=True,
        output_attentions=False,
    ):
        # 1st residual block
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (
            ngram_attention_output,
            self_attn_weights,
            self_attn_weights_ngram,
            present_key_value,
        ) = self.self_attn(
            hiddens=hiddens,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
        )
        hiddens = self.self_attn_layer_norm(hiddens + ngram_attention_output)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attn_weights = None
        if enc_hiddens is not None:
            # 2nd residual block
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hiddens=hiddens,
                key_value_states=enc_hiddens,
                attention_mask=encoder_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hiddens = self.cross_attn_layer_norm(attention_output + hiddens)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # 3rd residual block
        feed_forward_output = self.feed_forward(hiddens)
        hiddens = self.feed_forward_layer_norm(feed_forward_output + hiddens)

        outputs = (hiddens,)

        if output_attentions:
            outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

        if y_cache:
            outputs += (present_key_value,)

        return outputs


class Encoder(PreTrained):
    def __init__(self, config, word_embeddings: qc.Embed = None):
        super().__init__(config)

        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        )
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)
        self.embeddings_layer_norm = LayerNorm(config.d_model)

        self.layers = nn.ModuleList([EncLayer(config) for _ in range(config.num_encoder_layers)])

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

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds has to be passed.")
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("Make sure to only pass input_ids or inputs_embeds.")
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # prepare attention mask
        if attention_mask is not None:
            extended_attention_mask = (
                1.0
                - attention_mask[:, None, :].repeat(self.config.num_encoder_attention_heads, 1, 1)
            ) * -10000.0
            extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_attention_mask = None

        position_embeddings, position_ids = self.position_embeddings(
            inputs_embeds.shape[:2], inputs_embeds.device
        )

        hiddens = inputs_embeds + position_embeddings
        hiddens = self.embeddings_layer_norm(hiddens)
        hiddens = F.drop(hiddens, p=self.config.drop, training=self.training)

        enc_hiddens = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                enc_hiddens = enc_hiddens + (hiddens,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hiddens,
                    extended_attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hiddens,
                    attention_mask=extended_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hiddens = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            enc_hiddens = enc_hiddens + (hiddens,)

        if not return_dict:
            return tuple(v for v in [hiddens, enc_hiddens, all_attentions] if v is not None)
        return qo.Base(
            y=hiddens,
            hiddens=enc_hiddens,
            attns=all_attentions,
        )


class Decoder(PreTrained):
    def __init__(self, config, word_embeddings: qc.Embed = None):
        super().__init__(config)

        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.drop = config.drop
        self.max_target_positions = config.n_pos

        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        )
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)

        self.ngram_embeddings = qc.Embed(self.ngram, config.d_model, None)
        self.layers = nn.ModuleList([DecLayer(config) for _ in range(config.n_dec_lays)])
        self.embeddings_layer_norm = LayerNorm(config.d_model)

        self.gradient_checkpointing = False

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
        y_cache = y_cache if y_cache is not None else self.config.y_cache
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "Either `decoder_input_ids` or `decoder_inputs_embeds` has to be passed."
            )
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Make sure to only pass `decoder_input_ids` or `decoder_inputs_embeds`."
            )
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]

        main_stream_pos_embed, position_ids = self.position_embeddings(
            (batch_size, sequence_length),
            device=inputs_embeds.device,
            caches=caches,
        )

        if caches is not None:
            main_relative_position_buckets, predict_relative_position_buckets = None, None
        else:
            (
                main_relative_position_buckets,
                predict_relative_position_buckets,
            ) = self.compute_buffered_relative_buckets(position_ids)
        predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)

        # add position embeddings
        hiddens = inputs_embeds + main_stream_pos_embed

        ngram_embeddings = self.ngram_embeddings.weight

        # prepare attention mask
        if caches is not None:
            assert (
                hiddens.size(1) == 1
            ), "At the moment `y_cache` is only supported for `decoder_input_ids` of length 1"

            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).repeat(batch_size, 1, 1)
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = None
            extended_predict_attention_mask = None
        else:
            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed)
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = self.prepare_attention_mask(hiddens, attention_mask)
            extended_predict_attention_mask = self.prepare_predict_attention_mask(
                hiddens, attention_mask
            )

        # prepare encoder attention mask
        if encoder_attention_mask is not None:
            extended_encoder_attention_mask = (
                1.0
                - encoder_attention_mask[:, None, :].repeat(
                    self.config.num_decoder_attention_heads, 1, 1
                )
            ) * -10000.0
            extended_encoder_attention_mask = extended_encoder_attention_mask.to(
                inputs_embeds.dtype
            )
        else:
            extended_encoder_attention_mask = None

        hiddens = torch.cat([hiddens] + ngram_hidden_states, 1)

        if self.embeddings_layer_norm:
            hiddens = self.embeddings_layer_norm(hiddens)

        hiddens = F.drop(hiddens, p=self.drop, training=self.training)

        # init attns, hiddens and cache with empty tuples
        all_main_stream_hidden_states = () if output_hidden_states else None
        all_ngram_stream_hidden_states = (
            () if output_hidden_states and self.config.ngram > 0 else None
        )

        all_main_stream_attns = () if output_attentions else None
        all_ngram_stream_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and self.config.add_cross_attention else None
        present_key_values = () if y_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers))
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # grad cannot be kept because tensor is sliced
                all_main_stream_hidden_states += (hiddens[:, :sequence_length],)
                if self.config.ngram > 0:
                    all_ngram_stream_hidden_states += (hiddens[:, sequence_length:],)

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
                        return module(*inputs, y_cache, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hiddens,
                    extended_attention_mask,
                    enc_hiddens,
                    extended_encoder_attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    (cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    extended_predict_attention_mask,
                    main_relative_position_buckets,
                    predict_relative_position_buckets,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hiddens,
                    attention_mask=extended_attention_mask,
                    enc_hiddens=enc_hiddens,
                    encoder_attn_mask=extended_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    extended_predict_attention_mask=extended_predict_attention_mask,
                    main_relative_position_buckets=main_relative_position_buckets,
                    predict_relative_position_buckets=predict_relative_position_buckets,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    y_cache=y_cache,
                    output_attentions=output_attentions,
                )

            hiddens = layer_outputs[0]

            if y_cache:
                present_key_values += (layer_outputs[4 if output_attentions else 1],)

            if output_attentions:
                all_main_stream_attns += (layer_outputs[1],)
                all_ngram_stream_attns += (layer_outputs[2],)

                if self.config.add_cross_attention:
                    all_cross_attns += (layer_outputs[3],)

        if output_hidden_states:
            all_main_stream_hidden_states += (hiddens[:, :sequence_length],)
            if self.config.ngram > 0:
                all_ngram_stream_hidden_states += (hiddens[:, sequence_length:],)

        # split y for return
        y = hiddens[:, :sequence_length]
        last_hidden_state_ngram = hiddens[:, sequence_length:] if self.config.ngram > 0 else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    y,
                    last_hidden_state_ngram,
                    present_key_values,
                    all_main_stream_hidden_states,
                    all_ngram_stream_hidden_states,
                    all_main_stream_attns,
                    all_ngram_stream_attns,
                    all_cross_attns,
                ]
                if v is not None
            )
        return ProphetNetDecoderModelOutput(
            y=y,
            last_hidden_state_ngram=last_hidden_state_ngram,
            caches=present_key_values,
            hiddens=all_main_stream_hidden_states,
            hidden_states_ngram=all_ngram_stream_hidden_states,
            attns=all_main_stream_attns,
            ngram_attentions=all_ngram_stream_attns,
            crosses=all_cross_attns,
        )

    def compute_buffered_relative_buckets(self, position_ids):
        batch_size, sequence_length = position_ids.shape

        position_ids = (
            torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        )
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # buffer relative buckets
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(
            batch_size, 1, 1
        )
        predict_relative_buckets = torch.cat(
            [
                predict_relative_buckets[:, :sequence_length, :sequence_length],
                predict_relative_buckets[
                    :,
                    :sequence_length,
                    self.max_target_positions : self.max_target_positions + sequence_length,
                ],
            ],
            2,
        ).repeat(batch_size, 1, 1)

        return main_relative_buckets, predict_relative_buckets

    def prepare_attention_mask(self, hiddens, attention_mask):
        batch_size, seq_length = hiddens.shape[:2]
        causal_mask = torch.full(
            (seq_length, seq_length),
            -float("inf"),
            dtype=hiddens.dtype,
            device=hiddens.device,
        )
        causal_mask = torch.triu(causal_mask, 1)
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, :, :].expand(
            (batch_size,) + causal_mask.shape
        )
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, :]) * -10000.0
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return extended_attention_mask.repeat(self.config.num_decoder_attention_heads, 1, 1).to(
            hiddens.dtype
        )

    def prepare_predict_attention_mask(self, hiddens, attention_mask):
        batch_size, seq_length = hiddens.shape[:2]
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hiddens.device, hiddens.dtype
        )
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[
                    :,
                    :seq_length,
                    self.max_target_positions : self.max_target_positions + seq_length,
                ],
            ],
            dim=-1,
        )
        extended_predict_causal_mask = predict_causal_mask[:, None, :, :].expand(
            predict_causal_mask.shape[:1] + (batch_size,) + predict_causal_mask.shape[1:]
        )
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[None, :, None, :]) * -10000.0
            extended_attention_mask = extended_attention_mask.expand(
                (self.ngram, batch_size, seq_length, seq_length)
            )
            # predicted stream attention_mask should always be 0
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        return extended_predict_attention_mask.repeat(
            1, self.config.num_decoder_attention_heads, 1, 1
        ).to(hiddens.dtype)


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_enc_dec = False
        encoder_config.y_cache = False
        self.encoder = Encoder(encoder_config, self.word_embeddings)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_enc_dec = False
        self.decoder = Decoder(decoder_config, self.word_embeddings)

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
        y_cache = y_cache if y_cache is not None else self.config.y_cache
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # decoder outputs consists of (dec_features, caches, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            enc_hiddens=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            caches=caches,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            y_cache=y_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        return ProphetNetSeq2SeqModelOutput(
            y=decoder_outputs.y,
            last_hidden_state_ngram=decoder_outputs.last_hidden_state_ngram,
            caches=decoder_outputs.caches,
            hiddens=decoder_outputs.hiddens,
            decoder_ngram_hidden_states=decoder_outputs.hidden_states_ngram,
            attns=decoder_outputs.attns,
            decoder_ngram_attentions=decoder_outputs.ngram_attentions,
            crosses=decoder_outputs.crosses,
            enc_y=encoder_outputs.y,
            enc_hiddens=encoder_outputs.hiddens,
            enc_attns=encoder_outputs.attns,
        )


class ForCondGen(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.prophetnet = Model(config)
        self.padding_idx = config.PAD
        self.disable_ngram_loss = config.disable_ngram_loss

        self.lm_head = qc.Linear(config.d_model, config.s_vocab, bias=False)

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

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        outputs = self.prophetnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            caches=caches,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sequence_length = (
            decoder_input_ids.shape
            if decoder_input_ids is not None
            else decoder_inputs_embeds.shape[:2]
        )

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        # To use .view in loss computation, make sure that logits is contiguous.
        if not logits.is_contiguous():
            logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (
                (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
            )
        else:
            return ProphetNetSeq2SeqLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                caches=outputs.caches,
                hiddens=outputs.hiddens,
                decoder_ngram_hidden_states=outputs.decoder_ngram_hidden_states,
                attns=outputs.attns,
                decoder_ngram_attentions=outputs.decoder_ngram_attentions,
                crosses=outputs.crosses,
                enc_y=outputs.enc_y,
                enc_hiddens=outputs.enc_hiddens,
                enc_attns=outputs.enc_attns,
            )

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(
            ignore_index
        )

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        logits = logits.transpose(0, 1).contiguous()
        lprobs = F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        loss = F.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss


class ForCausal(PreTrained):
    def __init__(self, config):
        # set config for CLM
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_enc_dec = False
        super().__init__(config)
        self.prophetnet = ProphetNetDecoderWrapper(config)

        self.padding_idx = config.PAD
        self.disable_ngram_loss = config.disable_ngram_loss

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, caches, dec_hidden, dec_attn)
        outputs = self.prophetnet.decoder(
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

        batch_size, sequence_length = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        )

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (
                (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
            )
        else:
            return ProphetNetDecoderLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                caches=outputs.caches,
                hiddens=outputs.hiddens,
                hidden_states_ngram=outputs.hidden_states_ngram,
                attns=outputs.attns,
                ngram_attentions=outputs.ngram_attentions,
                crosses=outputs.crosses,
            )

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(
            ignore_index
        )

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        logits = logits.transpose(0, 1).contiguous()
        lprobs = F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        loss = F.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss


class ProphetNetDecoderWrapper(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = Decoder(config)

    def forward(self, *args, **kw):
        return self.decoder(*args, **kw)
