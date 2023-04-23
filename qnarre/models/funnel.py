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

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, FFNet, Masker, Pool
from ..prep.config.funnel import PreTrained


log = logging.get_logger(__name__)

from torch.nn import CrossEntropyLoss

LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

INF = 1e6


class FunnelEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.layer_norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.drop(embeddings)
        return embeddings


class FunnelAttentionStructure(qc.Module):
    cls_token_type_id = 2

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sin_dropout = qc.Dropout(config.drop)
        self.cos_dropout = qc.Dropout(config.drop)
        self.pooling_mult = None

    def init_attention_inputs(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
    ):
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.size(1)
        position_embeds = self.get_position_embeds(
            seq_len, inputs_embeds.dtype, inputs_embeds.device
        )
        token_type_mat = (
            self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        )
        cls_mask = (
            F.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        # Treat <cls> as in the same segment as both A & B
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        return cls_mat | token_type_mat

    def get_position_embeds(self, seq_len, dtype, device):
        d_model = self.config.d_model
        if self.config.attention_type == "factorized":
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            sinusoid = pos_seq[:, None] * inv_freq[None]
            sin_embed = torch.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = torch.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = torch.cat([sin_embed_d, sin_embed_d], dim=-1)
            psi = torch.cat([cos_embed, sin_embed], dim=-1)
            pi = torch.cat([cos_embed_d, cos_embed_d], dim=-1)
            omega = torch.cat([-sin_embed, cos_embed], dim=-1)
            return (phi, pi, psi, omega)
        else:
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype, device=device)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id[:, None] * inv_freq[None]
            sin_embed = self.sin_dropout(torch.sin(sinusoid))
            cos_embed = self.cos_dropout(torch.cos(sinusoid))
            pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)

            pos = torch.arange(0, seq_len, dtype=dtype, device=device)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config.num_blocks):
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos[:, None] + zero_offset
                    rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                    position_embeds_pooling = torch.gather(pos_embed, 0, rel_pos)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                position_embeds_no_pooling = torch.gather(pos_embed, 0, rel_pos)

                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index):
        if self.config.separate_cls:
            cls_pos = pos_id.new_tensor([-(2**block_index) + 1])
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_pool(self, tensor, axis):
        if tensor is None:
            return None
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)
        axis %= tensor.ndim
        axis_slice = (
            slice(None, -1, 2)
            if self.config.separate_cls and self.config.truncate_seq
            else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)
        return tensor[enc_slice]

    def pool_tensor(
        self,
        tensor,
        mode="mean",
        stride=2,
    ):
        if tensor is None:
            return None
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = F.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = F.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -F.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(self, output, attention_inputs):
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_heads, seq_len, max_rel_len = positional_attn.shape
    positional_attn = torch.reshape(positional_attn, [batch_size, n_heads, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = torch.reshape(
        positional_attn, [batch_size, n_heads, seq_len, max_rel_len - shift]
    )
    positional_attn = positional_attn[..., :context_len]
    return positional_attn


class FunnelRelMultiheadAttention(qc.Module):
    def __init__(self, config, block_index):
        super().__init__()
        self.config = config
        self.block_index = block_index
        d_model, n_heads, d_head = config.d_model, config.n_heads, config.d_head

        self.drop = qc.Dropout(config.drop)
        self.drop_attn = qc.Dropout(config.drop_attn)

        self.q_head = qc.Linear(d_model, n_heads * d_head, bias=False)
        self.k_head = qc.Linear(d_model, n_heads * d_head)
        self.v_head = qc.Linear(d_model, n_heads * d_head)

        self.r_w_bias = nn.Parameter(torch.zeros([n_heads, d_head]))
        self.r_r_bias = nn.Parameter(torch.zeros([n_heads, d_head]))
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_heads, d_head]))
        self.r_s_bias = nn.Parameter(torch.zeros([n_heads, d_head]))
        self.seg_embed = nn.Parameter(torch.zeros([2, n_heads, d_head]))

        self.post_proj = qc.Linear(n_heads * d_head, d_model)
        self.layer_norm = qc.LayerNorm(d_model, eps=config.eps)
        self.scale = 1.0 / (d_head**0.5)

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        if self.config.attention_type == "factorized":
            phi, pi, psi, omega = position_embeds
            # Shape n_heads x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_heads x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_heads x d_model
            q_r_attention = torch.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]

            # Shape batch_size x n_heads x seq_len x context_len
            positional_attn = torch.einsum("bind,jd->bnij", q_r_attention_1, psi) + torch.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_heads x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_heads x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_heads x d_model
            r_head = torch.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_heads x seq_len x max_rel_len
            positional_attn = torch.einsum("binh,tnh->bnit", q_head + v, r_head)
            # Shape batch_size x n_heads x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head has shape batch_size x seq_len x n_heads x d_head
        # Shape n_heads x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_heads x seq_len x 2
        token_type_bias = torch.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # Shape batch_size x n_heads x seq_len x context_len
        token_type_mat = token_type_mat[:, None].expand(
            [batch_size, q_head.shape[2], seq_len, context_len]
        )
        # Shapes batch_size x n_heads x seq_len
        diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
        # Shape batch_size x n_heads x seq_len x context_len
        token_type_attn = torch.where(
            token_type_mat,
            same_token_type.expand(token_type_mat.shape),
            diff_token_type.expand(token_type_mat.shape),
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(
        self,
        query,
        key,
        value,
        attention_inputs,
        output_attentions=False,
    ):
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_heads, d_head = self.config.n_heads, self.config.d_head

        # Shape batch_size x seq_len x n_heads x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_heads, d_head)
        # Shapes batch_size x context_len x n_heads x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_heads, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_heads, d_head)

        q_head = q_head * self.scale
        # Shape n_heads x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_heads x seq_len x context_len
        content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(
            position_embeds, q_head, context_len, cls_mask
        )
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        # perform masking
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.drop_attn(attn_prob)

        # attention output, shape batch_size x seq_len x n_heads x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_heads * d_head))
        attn_out = self.drop(attn_out)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class FunnelPositionwiseFFN(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = qc.Linear(config.d_model, config.d_inner)
        self.act = qu.activation(config.act)
        self.drop_act = qc.Dropout(config.drop_act)
        self.linear_2 = qc.Linear(config.d_inner, config.d_model)
        self.drop = qc.Dropout(config.drop)
        self.norm = qc.LayerNorm(config.d_model, config.eps)

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.act(h)
        h = self.drop_act(h)
        h = self.linear_2(h)
        h = self.drop(h)
        return self.norm(hidden + h)


class Layer(qc.Module):
    def __init__(self, config, block_index):
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(
        self,
        query,
        key,
        value,
        attention_inputs,
        output_attentions=False,
    ):
        attn = self.attention(
            query, key, value, attention_inputs, output_attentions=output_attentions
        )
        output = self.ffn(attn[0])
        return (output, attn[1]) if output_attentions else (output,)


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([Layer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_model, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for layer_index, layer in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_model
                        key = value = hidden if self.config.pool_q_only else pooled_model
                    else:
                        query = key = value = hidden
                    layer_output = layer(
                        query, key, value, attention_inputs, output_attentions=output_attentions
                    )
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(
                            attention_inputs
                        )

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return qo.Base(y=hidden, hiddens=all_hidden_states, attns=all_attentions)


def upsample(
    x,
    stride,
    target_len,
    separate_cls=True,
    truncate_seq=False,
):
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = torch.repeat_interleave(x, repeats=stride, dim=1)
    if separate_cls:
        if truncate_seq:
            output = F.pad(output, (0, 0, 0, stride - 1, 0, 0))
        output = output[:, : target_len - 1]
        output = torch.cat([cls, output], dim=1)
    else:
        output = output[:, :target_len]
    return output


class Decoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.ModuleList([Layer(config, 0) for _ in range(config.n_dec_lays)])

    def forward(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        upsampled_model = upsample(
            final_hidden,
            stride=2 ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config.separate_cls,
            truncate_seq=self.config.truncate_seq,
        )

        hidden = upsampled_model + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        for layer in self.layers:
            layer_output = layer(
                hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions
            )
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return qo.Base(y=hidden, hiddens=all_hidden_states, attns=all_attentions)


class FunnelDiscriminatorPredictions(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = qc.Linear(config.d_model, config.d_model)
        self.dense_prediction = qc.Linear(config.d_model, 1)

    def forward(self, discriminator_hidden_states):
        hiddens = self.dense(discriminator_hidden_states)
        hiddens = qu.activation(self.config.act)(hiddens)
        logits = self.dense_prediction(hiddens).squeeze()
        return logits


class FunnelBaseModel(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = Encoder(config)

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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
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

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.config.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        return qo.Base(
            y=decoder_outputs[0],
            hiddens=(encoder_outputs.hiddens + decoder_outputs.hiddens)
            if output_hidden_states
            else None,
            attns=(encoder_outputs.attns + decoder_outputs.attns) if output_attentions else None,
        )


class ForPreTraining(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.funnel = Model(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(
                    logits.view(-1, discriminator_sequence_output.shape[1]), labels.float()
                )

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return qo.WithLoss(
            loss=loss,
            logits=logits,
            hiddens=discriminator_hidden_states.hiddens,
            attns=discriminator_hidden_states.attns,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, **kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.funnel = FunnelBaseModel(config)
        self.classifier = Classifier(config, 1)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        y = outputs[0]
        pooled_output = y[:, 0]
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
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
        self.model = Model(**kw)
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
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
