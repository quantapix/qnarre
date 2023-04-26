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

import torch
import torch.utils.checkpoint
import math
import random

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


LIST = [
    "allenai/led-base-16384",
]


def shift_tokens_right(input_ids, PAD, decoder_start_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if PAD is None:
        raise ValueError("config.PAD has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, PAD)
    return shifted_input_ids


class LEDLearnedPositionalEmbedding(qc.Embed):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape, past_key_values_length=0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


# Copied from transformers.models.longformer.modeling_longformer.LongformerSelfAttention with Longformer->LEDEncoder
class LEDEncoderSelfAttention(qc.Module):
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
        query_vectors = self.query(hiddens)
        key_vectors = self.key(hiddens)
        value_vectors = self.value(hiddens)

        seq_len, batch_size, embed_dim = hiddens.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hiddens should have embed_dim = {self.embed_dim}, but has {embed_dim}"
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
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hiddens = hiddens.view(
            hiddens.size(0),
            hiddens.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hiddens.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
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
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * n_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )
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


class LEDEncoderAttention(qc.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.longformer_self_attn = LEDEncoderSelfAttention(config, layer_id=layer_id)
        self.output = qc.Linear(config.d_model, config.d_model)

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
        self_outputs = self.longformer_self_attn(
            hiddens=hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        attn_output = self.output(self_outputs[0])
        outputs = (attn_output,) + self_outputs[1:]

        return outputs


class LEDDecoderAttention(qc.Module):
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
        if self.head_dim * n_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by n_heads (got `embed_dim`: {self.embed_dim} and `n_heads`: {n_heads})."
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
        bsz, tgt_len, embed_dim = hiddens.size()

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
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
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

        attn_output = (
            attn_output.view(bsz, self.n_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class EncLayer(qc.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LEDEncoderAttention(config, layer_id)
        self.self_attn_layer_norm = qc.LayerNorm(self.embed_dim)
        self.drop = config.drop
        self.act = qu.activation(config.act)
        self.drop_act = config.drop_act
        self.fc1 = qc.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = qc.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = qc.LayerNorm(self.embed_dim)

    def forward(
        self,
        hiddens,
        attention_mask,
        layer_head_mask,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        residual = hiddens
        attn_outputs = self.self_attn(
            hiddens=hiddens,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        hiddens = attn_outputs[0]
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        hiddens = residual + hiddens
        hiddens = self.self_attn_layer_norm(hiddens)

        residual = hiddens
        hiddens = self.act(self.fc1(hiddens))
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
        return (hiddens,) + attn_outputs[1:]


class DecLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = LEDDecoderAttention(
            embed_dim=self.embed_dim,
            n_heads=config.decoder_attention_heads,
            drop=config.drop_attn,
            is_decoder=True,
        )
        self.drop = config.drop
        self.act = qu.activation(config.act)
        self.drop_act = config.drop_act
        self.self_attn_layer_norm = qc.LayerNorm(self.embed_dim)
        self.encoder_attn = LEDDecoderAttention(
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
        hiddens = self.act(self.fc1(hiddens))
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
        self.max_source_positions = config.max_encoder_position_embeddings

        if isinstance(config.attention_window, int):
            if config.attention_window % 2 != 0:
                raise ValueError("`config.attention_window` has to be an even value")
            if config.attention_window <= 0:
                raise ValueError("`config.attention_window` has to be positive")
            config.attention_window = [
                config.attention_window
            ] * config.n_lays  # one value per layer
        else:
            if len(config.attention_window) != config.n_lays:
                raise ValueError(
                    "`len(config.attention_window)` should equal `config.n_lays`. "
                    f"Expected {config.n_lays}, given {len(config.attention_window)}"
                )

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = qc.Embed(config.s_vocab, embed_dim, self.padding_idx)

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        self.layers = nn.ModuleList([EncLayer(config, i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = qc.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask, global_attention_mask):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        PAD,
    ):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        if attention_window % 2 != 0:
            raise ValueError(
                f"`attention_window` should be an even value. Given {attention_window}"
            )
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
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.PAD,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens

        return padding_len, input_ids, attention_mask, inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
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

        # check input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create default attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.size()[:-1], device=inputs_embeds.device, dtype=torch.long
            )

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        # pad input if necessary
        padding_len, input_ids, attention_mask, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            PAD=self.config.PAD,
        )

        # retrieve input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        # convert attention_mask to float
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)[:, 0, 0, :]

        # get masking tensors
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        embed_pos = self.embed_positions(input_shape)

        hiddens = inputs_embeds + embed_pos
        hiddens = self.layernorm_embedding(hiddens)
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hiddens,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, is_global_attn, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hiddens,
                        attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        is_index_masked,
                        is_index_global_attn,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hiddens,
                        attention_mask=attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
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

        if output_hidden_states:
            encoder_states = encoder_states + (hiddens,)

        # undo padding
        if padding_len > 0:
            # unpad `hiddens` because the calling function is expecting a length == input_ids.size(1)
            hiddens = hiddens[:, :-padding_len]
            if output_hidden_states:
                encoder_states = tuple([state[:, :-padding_len] for state in encoder_states])

            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])

        if not return_dict:
            return tuple(
                v
                for v in [hiddens, encoder_states, all_attentions, all_global_attentions]
                if v is not None
            )
        return qo.Base(
            y=hiddens,
            hiddens=encoder_states,
            attns=all_attentions,
            globals=all_global_attentions,
        )


class Decoder(PreTrained):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.drop = config.drop
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.PAD
        self.max_target_positions = config.max_decoder_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = qc.Embed(config.s_vocab, config.d_model, self.padding_idx)

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList([DecLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = qc.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
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
            inputs_embeds = self.embed_tokens(input_ids)

        # create causal mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = qu.causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
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
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if y_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
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
                    combined_attention_mask,
                    enc_hiddens,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hiddens,
                    attention_mask=combined_attention_mask,
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
        global_attention_mask=None,
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
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.PAD, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, qo.Base):
            encoder_outputs = qo.Base(
                y=encoder_outputs[0],
                hiddens=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attns=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                globals=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
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

        return qo.Seq2Seq(
            y=decoder_outputs.y,
            caches=decoder_outputs.caches,
            hiddens=decoder_outputs.hiddens,
            attns=decoder_outputs.attns,
            crosses=decoder_outputs.crosses,
            enc_y=encoder_outputs.y,
            enc_hiddens=encoder_outputs.hiddens,
            enc_attns=encoder_outputs.attns,
            enc_globals=encoder_outputs.globals,
        )


class ForCondGen(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.led = Model(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = qc.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)

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
        global_attention_mask=None,
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
            if y_cache:
                log.warning(
                    "The `y_cache` argument is changed to `False` since `labels` is provided."
                )
            y_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.PAD, self.config.decoder_start_token_id
                )

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
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

        return qo.LossSeq2Seq(
            loss=masked_lm_loss,
            logits=lm_logits,
            caches=outputs.caches,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
            enc_y=outputs.enc_y,
            enc_hiddens=outputs.enc_hiddens,
            enc_attns=outputs.enc_attns,
            enc_globals=outputs.enc_globals,
        )


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(cfg.d_model, "tanh", **kw)

    forward = qf.forward_seq

    def pre_proj(self, x, ys):
        eos_m = x.eq(self.cfg.EOS)
        y = ys[0]
        assert len(torch.unique_consecutive(eos_m.sum(1))) <= 1
        y = y[eos_m, :].view(y.size(0), -1, y.size(-1))
        return y[:, -1, :]


class ForTokClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class ForQA(PreTrained):
    def __init__(self, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
