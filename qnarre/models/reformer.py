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

import numpy as np
import sys
import torch
import torch.utils.checkpoint

from collections import namedtuple
from functools import reduce
from operator import mul
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.bert import PreTrained


from torch.autograd.function import Function
from torch.nn import CrossEntropyLoss

from ...pytorch_utils import apply_chunking_to_forward


log = logging.get_logger(__name__)


LIST = [
    "google/reformer-crime-and-punishment",
    "google/reformer-enwik8",
]


LSHSelfAttentionOutput = namedtuple(
    "LSHSelfAttentionOutput", ["hiddens", "attention_probs", "buckets"]
)
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hiddens", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput", ["hiddens", "attention_probs", "buckets"])
ReformerOutput = namedtuple(
    "ReformerOutput", ["hiddens", "attn_output", "attention_probs", "buckets"]
)
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hiddens", "grad_attn_output", "grad_model_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hiddens", "all_hidden_states", "all_attentions", "caches"],
)


def _stable_argsort(vector, dim):
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _get_min_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


class AxialPositionEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.drop = config.drop

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()

        if sum(self.axial_pos_embds_dim) != config.d_model:
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.d_model: {config.d_model}"
            )
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:])
            for weight in self.weights
        ]

        if self.training is True:
            if reduce(mul, self.axial_pos_shape) != sequence_length:
                raise ValueError(
                    f"If training, make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply to "
                    f"sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. "
                    f"You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} "
                    "or changing config.axial_pos_shape."
                )

            if self.drop > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                transposed_weights = weights.transpose(2, 1)
                # drop entire matrix of last two dims (prev dims 1 and 2)
                dropped_transposed_weights = F.dropout2d(
                    transposed_weights, p=self.drop, training=self.training
                )
                dropped_weights = dropped_transposed_weights.transpose(2, 1)

                position_encodings = torch.reshape(
                    dropped_weights, (batch_size, sequence_length, -1)
                )

            else:
                position_encodings = torch.cat(
                    [
                        torch.reshape(weight, (batch_size, sequence_length, -1))
                        for weight in broadcasted_weights
                    ],
                    dim=-1,
                )

        else:
            if reduce(mul, self.axial_pos_shape) < sequence_length:
                raise ValueError(
                    f"Make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply at least to "
                    f"max(sequence_length, least_common_mult_chunk_length): max({sequence_length}, "
                    f"{self.least_common_mult_chunk_length})."
                )

            # compute how many columns are needed
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) // self.axial_pos_shape[1])

            # cut to columns that are needed
            position_encodings = torch.cat(
                [weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights],
                dim=-1,
            )
            position_encodings = torch.reshape(
                position_encodings, (batch_size, -1, position_encodings.shape[-1])
            )

            # select correct position encodings
            position_encodings = torch.cat(
                [
                    torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
                    for i in range(batch_size)
                ],
                dim=0,
            )

        return position_encodings


class PositionEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.drop = config.drop
        self.embedding = qc.Embed(config.n_pos, config.d_model)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = F.drop(position_embeddings, p=self.drop, training=self.training)
        return position_embeddings


class ReformerEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.n_pos = config.n_pos
        self.drop = config.drop

        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model)
        self.position_embeddings = (
            AxialPositionEmbeddings(config)
            if config.axial_pos_embds
            else PositionEmbeddings(config)
        )

    def forward(
        self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                start_idx_pos_encodings,
                start_idx_pos_encodings + seq_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids.shape[-1] > self.n_pos:
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
                f"config.n_pos {self.n_pos}."
            )

        # drop
        embeddings = F.drop(inputs_embeds, p=self.drop, training=self.training)

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(
        self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None
    ):
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError(
                f"Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}"
            )


class LSHSelfAttention(qc.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.n_pos = config.n_pos

        self.drop = config.lsh_attention_probs_dropout_prob

        self.n_heads = config.n_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.n_heads * self.attention_head_size
        self.d_model = config.d_model

        # projection matrices
        self.query_key = qc.Linear(self.d_model, self.all_head_size, bias=False)
        self.value = qc.Linear(self.d_model, self.all_head_size, bias=False)

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        buckets=None,
        caches=None,
        y_cache=False,
        output_attentions=False,
        **kw,
    ):
        sequence_length = hiddens.shape[1]
        batch_size = hiddens.shape[0]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        do_cached_attention = y_cache and caches[1] is not None

        # check if cache shall be used and that hidden states are already cached
        if do_cached_attention:
            assert sequence_length == 1
            past_buckets = caches[0]
            past_states = caches[1]

            # get query vector
            query_vectors = self.query_key(hiddens)
            query_vectors = self._split_hidden_size_dim(
                query_vectors, self.n_heads, self.attention_head_size
            )

            if past_buckets is not None:
                (
                    key_value_hidden_states,
                    sorted_bucket_idx,
                    buckets,
                ) = self._get_relevant_hid_states_and_buckets(
                    query_vectors=query_vectors,
                    attention_mask=attention_mask,
                    num_hashes=num_hashes,
                    hiddens=hiddens,
                    past_states=past_states,
                    past_buckets=past_buckets,
                )

                query_key_vectors = self._query_per_attn_head(key_value_hidden_states)
                value_vectors = self._value_per_attn_head(key_value_hidden_states)

                # split key & value vectors by num hashes to apply
                # self attention on each separately
                query_key_vectors = self._split_seq_length_dim_to(
                    query_key_vectors,
                    num_hashes,
                    -1,
                    self.n_heads,
                    self.attention_head_size,
                )
                value_vectors = self._split_seq_length_dim_to(
                    value_vectors,
                    num_hashes,
                    -1,
                    self.n_heads,
                    self.attention_head_size,
                )
                # repeat query vectors across hash dimension
                query_vectors = query_vectors.unsqueeze(2).repeat(1, 1, num_hashes, 1, 1)
            else:
                key_value_hidden_states = torch.cat([past_states, hiddens], dim=1)

                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)

        else:
            # project hiddens to query_key and value
            query_vectors = None
            query_key_vectors = self.query_key(hiddens)
            value_vectors = self.value(hiddens)

        # if query key is not already split
        if not do_cached_attention or past_buckets is None:
            query_key_vectors = self._split_hidden_size_dim(
                query_key_vectors, self.n_heads, self.attention_head_size
            )
            value_vectors = self._split_hidden_size_dim(
                value_vectors, self.n_heads, self.attention_head_size
            )

        # cache buckets for next incremental decoding
        if (
            do_cached_attention
            and past_buckets is None
            and key_value_hidden_states.shape[1] >= self.chunk_length
        ):
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        # free memory
        del hiddens

        assert query_key_vectors.shape[-1] == self.attention_head_size
        assert value_vectors.shape[-1] == self.attention_head_size

        do_standard_self_attention = (sequence_length <= self.chunk_length) or (
            y_cache and caches[1] is not None
        )
        # LSH attention only makes sense if chunked attention should be performed
        if not do_standard_self_attention:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            # use cached buckets for backprop only
            if buckets is None:
                # hash query key vectors into buckets
                buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
            else:
                # make sure buckets has correct shape for LSH attention
                buckets = buckets.view(batch_size, self.n_heads, num_hashes * sequence_length)

            assert int(buckets.shape[-1]) == num_hashes * sequence_length

            (
                sorted_bucket_idx,
                undo_sorted_bucket_idx,
            ) = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(
                query_key_vectors, sorted_bucket_idx_per_hash, num_hashes
            )
            value_vectors = self._gather_by_expansion(
                value_vectors, sorted_bucket_idx_per_hash, num_hashes
            )
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.n_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.n_heads,
                self.attention_head_size,
            )

            if self.chunk_length is None:
                assert self.num_chunks_before == 0 and self.num_chunks_after == 0
        elif do_cached_attention and past_buckets is not None:
            # use max sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = torch.arange(
                sequence_length, device=query_key_vectors.device
            ).repeat(batch_size, self.n_heads, 1)

        # scale key vectors
        key_vectors = self._len_and_dim_norm(query_key_vectors)

        # set query_vectors to query key vectors if LSH self attention
        query_vectors = query_vectors if query_vectors is not None else query_key_vectors

        # free memory
        del query_key_vectors

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=head_mask,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=do_cached_attention,
        )

        # free memory
        del key_vectors, value_vectors

        # re-order out_vectors and logits
        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(
                out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx
            )

        if not do_standard_self_attention or (do_cached_attention and past_buckets is not None):
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.n_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.n_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.n_heads,
            sequence_length,
            self.attention_head_size,
        )

        out_vectors = self._merge_hidden_size_dims(
            out_vectors, self.n_heads, self.attention_head_size
        )

        if output_attentions is False:
            attention_probs = ()

        if buckets is not None:
            buckets = buckets.view(batch_size, self.n_heads, num_hashes, -1)

        return LSHSelfAttentionOutput(
            hiddens=out_vectors, attention_probs=attention_probs, buckets=buckets
        )

    def _query_per_attn_head(self, hiddens):
        per_head_query_key = self.query_key.weight.reshape(
            self.n_heads, self.attention_head_size, self.d_model
        ).transpose(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        query_key_vectors = torch.einsum("balh,ahr->balr", hiddens, per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hiddens):
        per_head_value = self.value.weight.reshape(
            self.n_heads, self.attention_head_size, self.d_model
        ).transpose(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        value_vectors = torch.einsum("balh,ahr->balr", hiddens, per_head_value)
        return value_vectors

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]
        if isinstance(self.num_buckets, int):
            assert self.num_buckets % 2 == 0
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (
            self.n_heads,
            vectors.shape[-1],
            num_hashes,
            rotation_size // 2,
        )
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[
                    ..., cur_sum : cur_sum + (bucket_factor // 2)
                ]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat(
                    [rotated_vectors_factor, -rotated_vectors_factor], dim=-1
                )
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (
            attention_mask.sum().item() < batch_size * attention_mask.shape[-1]
        ):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(
                buckets_mask,
                buckets,
                torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device),
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.n_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
        self, sequence_length, buckets, num_hashes
    ):
        # no gradients are needed
        with torch.no_grad():
            # hash-based sort
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2**num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.n_pos // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [
                2 ** (num_buckets_pow_2 // 2),
                2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2),
            ]

        log.warning(
            f"config.num_buckets is not set. Setting config.num_buckets to {num_buckets}..."
        )

        # set num buckets in config to be properly saved
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        do_standard_self_attention,
        do_cached_attention,
    ):
        # look at previous and following chunks if chunked attention
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(
                key_vectors, self.num_chunks_before, self.num_chunks_after
            )
            value_vectors = self._look_adjacent(
                value_vectors, self.num_chunks_before, self.num_chunks_after
            )

        # get logits and dots
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.n_heads
            )
            key_value_bucket_idx = self._look_adjacent(
                query_bucket_idx, self.num_chunks_before, self.num_chunks_after
            )
        elif do_cached_attention and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = (
                key_value_bucket_idx.new_ones(key_value_bucket_idx.shape[:-1] + (1,))
                * key_value_bucket_idx.max()
            )
        elif do_cached_attention and query_key_dots.ndim <= 4:
            query_bucket_idx = (query_key_dots.shape[-1] - 1) * torch.ones_like(query_key_dots)[
                :, :, :, -1
            ]
            key_value_bucket_idx = torch.arange(
                query_key_dots.shape[-1], dtype=torch.long, device=query_key_dots.device
            )[None, None, :].expand(query_bucket_idx.shape[:2] + (-1,))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        if not do_cached_attention:
            mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention,
            )

            if mask is not None:
                query_key_dots = torch.where(mask, query_key_dots, mask_value)

            # free memory
            del mask
        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        # apply self_mask
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # drop
        attention_probs = F.drop(attention_probs, p=self.drop, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self,
        query_indices,
        key_indices,
        attention_mask,
        query_key_dot_shape,
        do_standard_self_attention,
    ):
        # attention mask for LSH
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]
            if not do_standard_self_attention:
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # extract attention mask from LSH sorted key_indices
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(
                query_indices.device
            )

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hiddens, past_states, past_buckets
    ):
        # concat hidden states
        hiddens = torch.cat([past_states, hiddens], dim=1)

        # batch_size hidden
        batch_size = hiddens.shape[0]
        sequence_length = hiddens.shape[1]

        # check if cached buckets include pad bucket
        max_bucket = (
            self.num_buckets if isinstance(self.num_buckets, int) else reduce(mul, self.num_buckets)
        )

        # if pad bucket was cached => need to increase num buckets for caching
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1

        # retrieve query buckets
        query_buckets = self._hash_vectors(
            query_vectors, num_hashes, attention_mask, increase_num_buckets=increase_num_buckets
        )

        # concat buckets
        concat_buckets = torch.cat([past_buckets, query_buckets.unsqueeze(-1)], dim=-1)

        # hash-based sort
        bucket_idx = _stable_argsort(concat_buckets, dim=-1)

        # bucket_idx has shape: BatchSize x NumAttnHeads x NumHashes x SequenceLength
        assert bucket_idx.shape == (
            batch_size,
            self.n_heads,
            num_hashes,
            sequence_length,
        )

        # find indices of new bucket indices
        relevant_bucket_idx = (bucket_idx == (bucket_idx.shape[-1] - 1)).nonzero()

        # expand relevant bucket indices to its chunks
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(
            relevant_bucket_idx, sequence_length
        )
        relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.transpose(0, 1))]

        # adapt bucket_idx for batch and hidden states for index select
        bucket_idx_batch_offset = sequence_length * (
            batch_size
            * torch.arange(
                relevant_bucket_idx_chunk.shape[-1], device=hiddens.device, dtype=torch.long
            )
            // relevant_bucket_idx_chunk.shape[-1]
        )

        # add batch offset
        relevant_bucket_idx_chunk_all_batch = relevant_bucket_idx_chunk + bucket_idx_batch_offset
        hiddens = hiddens.reshape((-1, self.d_model))

        # select all relevant hidden states
        relevant_hidden_states = hiddens.index_select(0, relevant_bucket_idx_chunk_all_batch)

        # reshape hidden states and bucket_idx to correct output
        relevant_hidden_states = relevant_hidden_states.reshape(
            batch_size, self.n_heads, -1, self.d_model
        )
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(
            batch_size, self.n_heads, num_hashes, -1
        )

        assert (
            relevant_hidden_states.shape[2]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes
        )

        assert (
            relevant_bucket_idx_chunk.shape[-1]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length
        )

        return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        # get relevant indices of where chunk starts and its size
        start_indices_chunk = (
            (indices[:, -1] // self.chunk_length) - self.num_chunks_before
        ) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        # expand start indices and add correct chunk offset via arange
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(
            indices.shape[0], total_chunk_size
        )
        chunk_sequence_indices = expanded_start_indices + torch.arange(
            total_chunk_size, device=indices.device, dtype=torch.long
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        # make sure that circular logic holds via % seq len
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        # expand indices and set indices correctly
        indices = (
            indices.unsqueeze(1)
            .expand((indices.shape[0], total_chunk_size, -1))
            .flatten(0, 1)
            .clone()
        )
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors):
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        variance = torch.mean(x**2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class ReverseSort(Function):
    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            # undo sort to have correct order for next layer
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(
                out_vectors.shape
            )
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None


class LocalSelfAttention(qc.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.PAD = config.PAD

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.n_heads * self.attention_head_size
        self.d_model = config.d_model

        # projection matrices
        self.query = qc.Linear(self.d_model, self.all_head_size, bias=False)
        self.key = qc.Linear(self.d_model, self.all_head_size, bias=False)
        self.value = qc.Linear(self.d_model, self.all_head_size, bias=False)

        self.drop = config.local_attention_probs_dropout_prob

        # save mask value here
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        caches=None,
        y_cache=False,
        output_attentions=False,
        **kw,
    ):
        sequence_length = hiddens.shape[1]
        batch_size = hiddens.shape[0]

        # check if cache shall be used and that hidden states are already cached
        if y_cache and caches[1] is not None:
            assert caches[0] is None
            key_value_hidden_states = self._retrieve_relevant_hidden_states(
                caches[1], self.chunk_length, self.num_chunks_before
            )
            key_value_hidden_states = torch.cat([key_value_hidden_states, hiddens], dim=1)

            # only query vector for last token
            query_vectors = self.query(hiddens)
            # compute key and value for relevant chunk
            key_vectors = self.key(key_value_hidden_states)
            value_vectors = self.value(key_value_hidden_states)

            # free memory
            del key_value_hidden_states
        else:
            # project hiddens to query, key and value
            query_vectors = self.query(hiddens)
            key_vectors = self.key(hiddens)
            value_vectors = self.value(hiddens)

        # split last dim into `config.n_heads` and `config.attention_head_size`
        query_vectors = self._split_hidden_size_dim(
            query_vectors, self.n_heads, self.attention_head_size
        )
        key_vectors = self._split_hidden_size_dim(
            key_vectors, self.n_heads, self.attention_head_size
        )
        value_vectors = self._split_hidden_size_dim(
            value_vectors, self.n_heads, self.attention_head_size
        )

        assert query_vectors.shape[-1] == self.attention_head_size
        assert key_vectors.shape[-1] == self.attention_head_size
        assert value_vectors.shape[-1] == self.attention_head_size

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0

        # normalize key vectors
        key_vectors = key_vectors / torch.sqrt(
            torch.tensor(
                self.attention_head_size, device=key_vectors.device, dtype=key_vectors.dtype
            )
        )

        # get sequence length indices
        indices = torch.arange(sequence_length, device=query_vectors.device).repeat(
            batch_size, self.n_heads, 1
        )

        # if one should do normal n^2 self-attention
        do_standard_self_attention = sequence_length <= self.chunk_length

        # if input should be chunked
        if not do_standard_self_attention:
            # chunk vectors
            # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
            query_vectors = self._split_seq_length_dim_to(
                query_vectors,
                -1,
                self.chunk_length,
                self.n_heads,
                self.attention_head_size,
            )
            key_vectors = self._split_seq_length_dim_to(
                key_vectors,
                -1,
                self.chunk_length,
                self.n_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.n_heads,
                self.attention_head_size,
            )

            # chunk indices
            query_indices = self._split_seq_length_dim_to(
                indices, -1, self.chunk_length, self.n_heads
            )
            key_indices = self._split_seq_length_dim_to(
                indices, -1, self.chunk_length, self.n_heads
            )

            # append chunks before and after
            key_vectors = self._look_adjacent(
                key_vectors, self.num_chunks_before, self.num_chunks_after
            )
            value_vectors = self._look_adjacent(
                value_vectors, self.num_chunks_before, self.num_chunks_after
            )
            key_indices = self._look_adjacent(
                key_indices, self.num_chunks_before, self.num_chunks_after
            )
        else:
            query_indices = key_indices = indices

        # query-key matmul: QK^T
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(
            query_indices,
            key_indices,
            attention_mask,
            query_key_dots.shape,
            do_standard_self_attention,
        )

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16.half()
            else:
                mask_value = self.mask_value_float32

            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # softmax
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del logits

        # drop
        attention_probs = F.drop(attention_probs, p=self.drop, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if not do_standard_self_attention:
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        assert out_vectors.shape == (
            batch_size,
            self.n_heads,
            sequence_length,
            self.attention_head_size,
        )

        out_vectors = self._merge_hidden_size_dims(
            out_vectors, self.n_heads, self.attention_head_size
        )

        if output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hiddens=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(
        self,
        query_indices,
        key_indices,
        attention_mask,
        query_key_dots_shape,
        do_standard_self_attention,
    ):

        # chunk attention mask and look before and after
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]

            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(
                    attention_mask, -1, self.chunk_length, 1
                )
                attention_mask = self._look_adjacent(
                    attention_mask, self.num_chunks_before, self.num_chunks_after
                )
            # create attn_mask
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(
                query_indices.device
            )

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length, num_chunks_before):
        start_position = (
            (previous_hidden_states.shape[1] // chunk_length) - num_chunks_before
        ) * chunk_length
        return previous_hidden_states[:, start_position:]


class ReformerSelfOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.n_heads * config.attention_head_size
        self.drop = config.drop

        self.dense = qc.Linear(all_head_size, config.d_model, bias=False)

    def forward(self, hiddens):
        hiddens = self.dense(hiddens)
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)
        return hiddens


class Attention(qc.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers

        self.layer_norm = qc.LayerNorm(config.d_model, eps=config.eps)

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(["lsh", "local"]):
            # get correct attn layers
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        self.output = ReformerSelfOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        caches=None,
        y_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        buckets=None,
    ):
        hiddens = self.layer_norm(hiddens)

        # make sure cached hidden states is set to None for backward pass
        if caches is not None:
            caches_layer = caches[self.layer_id]
        else:
            caches_layer = None

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hiddens=hiddens,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            caches=caches_layer,
            y_cache=y_cache,
            output_attentions=output_attentions,
            buckets=buckets,
        )

        # add buckets if necessary
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        # cache hidden states for future use
        if y_cache:
            if caches[self.layer_id][0] is None:
                # padded input should not be cached
                past_buckets = (
                    buckets[:, :, :, :orig_sequence_length]
                    if (buckets is not None and orig_sequence_length > 1)
                    else buckets
                )
            else:
                past_buckets = torch.cat([caches[self.layer_id][0], buckets], dim=-1)

            if caches[self.layer_id][1] is None:
                # padded input should not be cached
                past_states = hiddens[:, :orig_sequence_length]
            else:
                past_states = torch.cat([caches[self.layer_id][1], hiddens], dim=1)

            caches[self.layer_id] = (past_buckets, past_states)
        # compute attention feed forward output
        attention_output = self.output(self_attention_outputs.hiddens)

        return AttentionOutput(
            hiddens=attention_output,
            attention_probs=self_attention_outputs.attention_probs,
            buckets=buckets,
        )


class ReformerFeedForwardDense(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.drop = cfg.drop
        self.act = qu.activation(cfg.act)
        self.dense = qc.Linear(cfg.d_model, cfg.feed_forward_size)

    def forward(self, x):
        y = self.dense(x)
        y = F.drop(y, p=self.drop, training=self.training)
        y = self.act(y)
        return y


class ReformerFeedForwardOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.drop = config.drop
        self.dense = qc.Linear(config.feed_forward_size, config.d_model)

    def forward(self, x):
        y = self.dense(x)
        y = F.drop(y, p=self.drop, training=self.training)
        return y


class ChunkReformerFeedForward(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hiddens):
        hiddens = self.layer_norm(hiddens)
        hiddens = self.dense(hiddens)
        return self.output(hiddens)


class Layer(qc.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = Attention(config, layer_id)
        # drop requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.feed_forward_seed)

    def forward(
        self,
        prev_attn_output,
        hiddens,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        caches=None,
        y_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
    ):
        with torch.no_grad():
            if self.training:
                self._init_attention_seed()

            attn_outputs = self.attention(
                hiddens=hiddens,
                head_mask=head_mask,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                caches=caches,
                y_cache=y_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs.hiddens

            # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # free memory
            del prev_attn_output

            # every forward pass we sample a different seed
            # for drop and save seed for forward fn in backward
            # to have correct drop
            if self.training:
                self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hiddens = hiddens + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hiddens=hiddens,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    def backward_pass(
        self,
        next_attn_output,
        hiddens,
        grad_attn_output,
        grad_model_states,
        attention_mask=None,
        head_mask=None,
        buckets=None,
    ):
        assert self.training

        with torch.enable_grad():
            next_attn_output.requires_grad = True

            # set seed to have correct drop
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_model_states, retain_graph=True)

        with torch.no_grad():
            # X_2 = Y_2 - g(Y_1)
            hiddens = hiddens - res_hidden_states
            del res_hidden_states

            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None

        with torch.enable_grad():
            hiddens.requires_grad = True

            # set seed to have correct drop
            torch.manual_seed(self.attention_seed)
            # f(X_2)
            # use cached buckets for backprob if buckets not None for LSHSelfAttention
            output = self.attention(
                hiddens=hiddens,
                head_mask=head_mask,
                attention_mask=attention_mask,
                buckets=buckets,
            ).hiddens
            output.backward(grad_attn_output, retain_graph=True)

        with torch.no_grad():
            # X_1 = Y_1 - f(X_2)
            attn_output = next_attn_output - output
            del output, next_attn_output

            grad_model_states = grad_model_states + hiddens.grad
            hiddens.grad = None
            hiddens = hiddens.detach()

        return ReformerBackwardOutput(
            attn_output=attn_output,
            hiddens=hiddens,
            grad_attn_output=grad_attn_output,
            grad_model_states=grad_model_states,
        )


class _ReversibleFunction(Function):
    @staticmethod
    def forward(
        ctx,
        hiddens,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        caches,
        y_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hiddens, attn_output = torch.chunk(hiddens, 2, dim=-1)

        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hiddens)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hiddens=hiddens,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                caches=caches,
                y_cache=y_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )

            attn_output = layer_outputs.attn_output
            hiddens = layer_outputs.hiddens
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hiddens)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hiddens.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hiddens], dim=-1)

    @staticmethod
    def backward(ctx, grad_model_states):
        grad_attn_output, grad_model_states = torch.chunk(grad_model_states, 2, dim=-1)

        # retrieve params from ctx for backward
        attn_output, hiddens = ctx.saved_tensors

        # create tuple
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hiddens=hiddens,
            grad_attn_output=grad_attn_output,
            grad_model_states=grad_model_states,
        )

        # free memory
        del grad_attn_output, grad_model_states, attn_output, hiddens

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            # pop last buckets from stack
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hiddens=output.hiddens,
                grad_attn_output=output.grad_attn_output,
                grad_model_states=output.grad_model_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )

        assert all_buckets == (), "buckets have to be empty after backpropagation"
        grad_model_states = torch.cat([output.grad_attn_output, output.grad_model_states], dim=-1)

        # num of return vars has to match num of forward() args
        # return gradient for hiddens arg and None for other args
        return grad_model_states, None, None, None, None, None, None, None, None, None, None, None


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.drop = config.drop

        self.layers = nn.ModuleList([Layer(config, i) for i in range(config.n_lays)])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * d_model
        self.layer_norm = qc.LayerNorm(2 * config.d_model, eps=config.eps)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        caches=None,
        y_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # hiddens and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # init cached hidden states if necessary
        if caches is None:
            caches = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hiddens = torch.cat([hiddens, hiddens], dim=-1)
        hiddens = _ReversibleFunction.apply(
            hiddens,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            caches,
            y_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hiddens = self.layer_norm(hiddens)

        # Apply drop
        hiddens = F.drop(hiddens, p=self.drop, training=self.training)

        return ReformerEncoderOutput(
            hiddens=hiddens,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            caches=caches,
        )


class ReformerOnlyLMHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        self.decoder = qc.Linear(2 * config.d_model, config.s_vocab, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.s_vocab))
        self.decoder.bias = self.bias

    def forward(self, hiddens):
        return apply_chunking_to_forward(
            self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hiddens
        )

    def forward_chunk(self, hiddens):
        hiddens = self.decoder(hiddens)
        return hiddens

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        assert self.config.n_lays > 0
        self.embeddings = ReformerEmbeddings(config)
        self.encoder = Encoder(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        caches=None,
        y_cache=None,
        output_hidden_states=None,
        output_attentions=None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # noqa: F841
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # noqa: F841
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert len(input_shape) == 2

        if caches is not None:
            assert not self.training

        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_lays, is_attention_chunked=True)

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        min_chunk_length = _get_min_chunk_len(self.config)

        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0
            and input_shape[-1] > min_chunk_length
            and caches is None
        )

        if must_pad_to_match_chunk_length:
            padding_length = (
                least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length
            )

            if self.training is True:
                raise ValueError(
                    f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                    f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                    f"of {input_shape[-1] + padding_length}."
                )

            # pad input
            (
                input_ids,
                inputs_embeds,
                attention_mask,
                position_ids,
                input_shape,
            ) = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )

        # start index for position encoding depends on incremental decoding
        if caches is not None:
            start_idx_pos_encodings = caches[0][1].shape[1]
        else:
            start_idx_pos_encodings = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )

        encoder_outputs = self.encoder(
            hiddens=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            caches=caches,
            y_cache=y_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.hiddens

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        caches = encoder_outputs.caches if y_cache else None
        hiddens = encoder_outputs.all_hidden_states if output_hidden_states else None
        attns = encoder_outputs.all_attentions if output_attentions else None

        if not return_dict:
            return tuple(v for v in [sequence_output, caches, hiddens, attns] if v is not None)
        return qo.WithCaches(
            y=sequence_output,
            caches=caches,
            hiddens=hiddens,
            attns=attns,
        )

    def _pad_to_mult_of_chunk_length(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        input_shape=None,
        padding_length=None,
        padded_seq_length=None,
        device=None,
    ):
        log.info(
            f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
            f"multiple of `config.chunk_length`: {padded_seq_length}"
        )

        padded_input_ids = torch.full(
            (input_shape[0], padding_length),
            self.config.PAD,
            device=device,
            dtype=torch.long,
        )

        # Extend `attention_mask`
        if attention_mask is not None:
            pad_attention_mask = torch.zeros(
                input_shape[0], padding_length, device=device, dtype=attention_mask.dtype
            )

            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.uint8),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.uint8),
                ],
                dim=-1,
            )

        # Extend `input_ids` with padding to match least common multiple chunk_length
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()

            # Pad position ids if given
            if position_ids is not None:
                padded_position_ids = torch.arange(
                    input_shape[-1], padded_seq_length, dtype=torch.long, device=device
                )
                padded_position_ids = position_ids.unsqueeze(0).expand(
                    input_shape[0], padding_length
                )
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)

        # Extend `inputs_embeds` with padding to match least common multiple chunk_length
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape


class ReformerModelWithLMHead(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        assert config.is_decoder
        assert "local" not in self.config.attn_layers or config.local_num_chunks_after == 0
        assert "lsh" not in self.config.attn_layers or config.lsh_num_chunks_after == 0

        self.reformer = Model(config)
        self.lm_head = ReformerOnlyLMHead(config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        caches=None,
        y_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            caches=caches,
            y_cache=y_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.s_vocab), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCaches(
            loss=loss,
            logits=logits,
            caches=reformer_outputs.caches,
            hiddens=reformer_outputs.hiddens,
            attns=reformer_outputs.attns,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        assert not cfg.is_decoder
        self.model = Model(**kw)
        self.proj = ReformerOnlyLMHead(**kw)

    forward = qf.forward_masked


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, "tanh", **kw, d_model=2 * cfg.d_model)

    forward = qf.forward_seq


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
