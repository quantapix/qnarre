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
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)

from torch.nn import CrossEntropyLoss

from ...pytorch_utils import (
    apply_chunking_to_forward,
)


LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
]


def _get_question_end_index(input_ids, sep_token_id):
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    assert (
        sep_token_indices.shape[0] == 3 * batch_size
    ), f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]


def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    # bool attention mask with True in locations of global attention
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.uint8)
    else:
        # last token is separation token and should not be counted and in the middle are two separation tokens
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(
            torch.uint8
        ) * (attention_mask.expand_as(input_ids) < input_ids.shape[-1]).to(torch.uint8)

    return attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class LongformerEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)

        # self.norm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.pos_type = getattr(config, "pos_type", "absolute")

        self.padding_idx = config.PAD
        self.position_embeddings = qc.Embed(
            config.n_pos, config.d_model, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings

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


class LongformerSelfAttention(qc.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_heads})"
            )
        self.n_heads = config.n_heads
        self.head_dim = int(config.d_model / config.n_heads)
        self.embed_dim = config.d_model

        self.query = qc.Linear(config.d_model, self.embed_dim)
        self.key = qc.Linear(config.d_model, self.embed_dim)
        self.value = qc.Linear(config.d_model, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = qc.Linear(config.d_model, self.embed_dim)
        self.key_global = qc.Linear(config.d_model, self.embed_dim)
        self.value_global = qc.Linear(config.d_model, self.embed_dim)

        self.drop = config.drop_attn

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
        self,
        hiddens,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        hiddens = hiddens.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hiddens)
        key_vectors = self.key(hiddens)
        value_vectors = self.value(hiddens)

        seq_len, batch_size, embed_dim = hiddens.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hiddens should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(
            seq_len, batch_size, self.n_heads, self.head_dim
        ).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.n_heads, self.head_dim).transpose(
            0, 1
        )

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.n_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.n_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, n_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = F.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.n_heads,
            ), f"Head mask for a single layer should be of size {(self.n_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply drop
        attn_probs = F.drop(attn_probs, p=self.drop, training=self.training)

        value_vectors = value_vectors.view(
            seq_len, batch_size, self.n_heads, self.head_dim
        ).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ), "Unexpected size"
        attn_output = (
            attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        )

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hiddens=hiddens,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2],
            hidden_states_padded.size(-1),
            hidden_states_padded.size(-2),
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_model_states):
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_model_states.size()
        chunked_model_states = F.pad(
            chunked_model_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_model_states = chunked_model_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_model_states = chunked_model_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_model_states = chunked_model_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_model_states = chunked_model_states[:, :, :, :-1]
        return chunked_model_states

    @staticmethod
    def _chunk(hiddens, window_overlap):
        hiddens = hiddens.view(
            hiddens.size(0),
            hiddens.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hiddens.size(2),
        )
        chunk_size = list(hiddens.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hiddens.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hiddens.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = (
            input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        )
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(
            beginning_mask == 1, -float("inf")
        )  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(
            ending_mask == 1, -float("inf")
        )  # `== 1` converts to bool or uint8

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        batch_size, seq_len, n_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and n_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * n_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * n_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (query, key)
        )  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * n_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attns
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[
            :, 0, 1:window_overlap, 1:window_overlap
        ] = diagonal_chunked_attention_scores[:, 0, : window_overlap - 1, 1 - window_overlap :]

        # separate batch_size and n_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, n_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        batch_size, seq_len, n_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and n_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * n_heads,
            seq_len // window_overlap,
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and n_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * n_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (
            batch_size * n_heads,
            chunks_count + 1,
            3 * window_overlap,
            head_dim,
        )
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(
            size=chunked_value_size, stride=chunked_value_stride
        )

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, n_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(
            as_tuple=True
        )
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.n_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[
            is_index_global_attn_nonzero
        ]

        # (batch_size, seq_len, n_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum(
            "blhd,bshd->blhs", (query_vectors, key_vectors_only_global)
        )

        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.n_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[
            is_index_global_attn_nonzero
        ]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(),
            value_vectors_only_global.transpose(1, 2).clone(),
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hiddens,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        seq_len, batch_size = hiddens.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hiddens.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim
        )
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hiddens[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hiddens)
        global_value_vectors = self.value_global(hiddens)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.n_heads, self.head_dim)
            .transpose(0, 1)
        )  # (batch_size * self.n_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous()
            .view(-1, batch_size * self.n_heads, self.head_dim)
            .transpose(0, 1)
        )  # batch_size * self.n_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous()
            .view(-1, batch_size * self.n_heads, self.head_dim)
            .transpose(0, 1)
        )  # batch_size * self.n_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(
            global_query_vectors_only_global, global_key_vectors.transpose(1, 2)
        )

        assert list(global_attn_scores.size()) == [
            batch_size * self.n_heads,
            max_num_global_attn_indices,
            seq_len,
        ], f"global_attn_scores have the wrong size. Size should be {(batch_size * self.n_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}."

        global_attn_scores = global_attn_scores.view(
            batch_size, self.n_heads, max_num_global_attn_indices, seq_len
        )

        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            -10000.0,
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.n_heads, max_num_global_attn_indices, seq_len
        )

        # compute global attn probs
        global_attn_probs_float = F.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.n_heads,
            ), f"Head mask for a single layer should be of size {(self.n_heads,)}, but is {layer_head_mask.size()}"
            global_attn_probs_float = layer_head_mask.view(
                1, -1, 1, 1
            ) * global_attn_probs_float.view(
                batch_size, self.n_heads, max_num_global_attn_indices, seq_len
            )
            global_attn_probs_float = global_attn_probs_float.view(
                batch_size * self.n_heads, max_num_global_attn_indices, seq_len
            )

        global_attn_probs = F.drop(
            global_attn_probs_float.type_as(global_attn_scores),
            p=self.drop,
            training=self.training,
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.n_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.n_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}."

        global_attn_probs = global_attn_probs.view(
            batch_size, self.n_heads, max_num_global_attn_indices, seq_len
        )
        global_attn_output = global_attn_output.view(
            batch_size, self.n_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output, global_attn_probs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LongformerSelfOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_model, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


class Attention(qc.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.self = LongformerSelfAttention(config, layer_id)
        self.output = LongformerSelfOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hiddens)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LongformerIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LongformerOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_ff, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


class Layer(qc.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = Attention(config, layer_id)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        hiddens,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Layer(config, layer_id=i) for i in range(config.n_lays)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attns.
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, is_global_attn, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hiddens,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    is_index_masked,
                    is_index_global_attn,
                )
            else:
                layer_outputs = layer_module(
                    hiddens,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                    output_attentions=output_attentions,
                )
            hiddens = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (
                        layer_outputs[2].transpose(2, 3),
                    )

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        # undo padding
        if padding_len > 0:
            # unpad `hiddens` because the calling function is expecting a length == input_ids.size(1)
            hiddens = hiddens[:, :-padding_len]
            if output_hidden_states:
                all_hidden_states = tuple([state[:, :-padding_len] for state in all_hidden_states])

            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])

        if not return_dict:
            return tuple(
                v
                for v in [hiddens, all_hidden_states, all_attentions, all_global_attentions]
                if v is not None
            )
        return qo.Base(
            y=hiddens,
            hiddens=all_hidden_states,
            attns=all_attentions,
            globals=all_global_attentions,
        )


class Model(PreTrained):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0
            assert config.attention_window > 0
            config.attention_window = [
                config.attention_window
            ] * config.n_lays  # one value per layer
        else:
            assert len(config.attention_window) == config.n_lays
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = Encoder(config)
        self.pooler = Pool(config) if add_pooling_layer else None

    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds,
        PAD,
    ):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            log.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=PAD)
            if position_ids is not None:
                # pad with position_id = PAD as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=PAD)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.PAD,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = F.pad(
                token_type_ids, (0, padding_len), value=0
            )  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask, global_attention_mask):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        ) = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            PAD=self.config.PAD,
        )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )[:, 0, 0, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return qo.WithPools(
            y=sequence_output,
            pools=pooled_output,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            globals=encoder_outputs.globals,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(**kw)

    forward = qf.forward_masked


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(cfg.d_model, "tanh", **kw)

    def forward(self, x, g_mask=None, **kw):
        if g_mask is None:
            g_mask = torch.zeros_like(x)
            g_mask[:, 0] = 1
        return qf.forward_seq(x, g_mask=g_mask, **kw)


class ForTokClass(PreTrained):
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

    def forward(self, x, globals=None, **kw):
        if globals is None:
            assert x is not None
        globals = _compute_global_attention_mask(x, self.cfg.sep_token_id)
        return qf.forward_qa(x, globals=globals, **kw)


class ForMulti(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.longformer = Model(config)
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        labels=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None and input_ids is not None:
            log.info("Initializing global attention on multiple choice...")
            global_attention_mask = torch.stack(
                [
                    _compute_global_attention_mask(
                        input_ids[:, i], self.config.sep_token_id, before_sep_token=False
                    )
                    for i in range(num_choices)
                ],
                dim=1,
            )

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
        flat_global_attention_mask = (
            global_attention_mask.view(-1, global_attention_mask.size(-1))
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
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
            globals=outputs.globals,
        )
