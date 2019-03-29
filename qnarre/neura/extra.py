# Copyright 2019 Quantapix Authors. All Rights Reserved.
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

import tensorflow as tf

import qneura.quess as q
from . import quess_utils as qu

ks = tf.keras
K = ks.backend
kls = ks.layers


class DotProductRelativeAttn(q.Attention):
    def attention(self, q, k, v, a, **kw):
        PS = self.PS
        return qu.dot_product_attention_relative(
            q,
            k,
            v,
            a,
            PS.max_position,
            dropout=PS.attn_drop,
        )


class DotProductUnmaskedRelativeAttn(q.Attention):
    def attention(self, q, k, v, a, **kw):
        PS = self.PS
        return qu.dot_product_unmasked_self_attention_relative_v2(
            q,
            k,
            v,
            a,
            PS.max_position,
            dropout=PS.attn_drop,
            broadcast_dims=PS.attn_bdims,
            heads_share_relative_embedding=PS.heads_share_embed,
            add_relative_to_values=PS.add_relative)


class DotProductRelativeV2Attn(q.Attention):
    def attention(self, q, k, v, a, **kw):
        PS = self.PS
        return qu.dot_product_self_attention_relative_v2(
            q,
            k,
            v,
            a,
            PS.max_position,
            dropout=PS.attn_drop,
            broadcast_dims=PS.attn_bdims,
            heads_share_relative_embedding=PS.heads_share_embed,
            add_relative_to_values=PS.add_relative)


class LocalWithinBlockMaskRightAttn(q.Attention):
    block_length = 128

    def attention(self, q, k, v, a, **kw):
        return qu.masked_within_block_local_attention_1d(
            q, k, v, block_length=self.block_length)


class LocalRelativeMaskRightAttn(q.Attention):
    block_length = 128

    def attention(self, q, k, v, a, **kw):
        PS = self.PS
        return qu.masked_relative_local_attention_1d(
            q,
            k,
            v,
            block_length=self.block_length,
            dropout=PS.attn_drop,
            heads_share_relative_embedding=PS.heads_share_embed,
            add_relative_to_values=PS.add_relative)


class LocalMaskRightAttn(q.Attention):
    block_length = 128

    def attention(self, q, k, v, a, **kw):
        return qu.masked_local_attention_1d(
            q, k, v, block_length=self.block_length)


class LocalUnmaskedAttn(q.Attention):
    block_length = 128
    block_width = 128

    def attention(self, q, k, v, a, **kw):
        return qu.local_attention_1d(
            q, k, v, block_length=self.block_length, f_width=self.block_width)


class MaskedDilatedAttn(q.Attention):
    num_memory_blocks = 2
    block_length = 128
    block_width = 128
    gap_size = 0

    def attention(self, q, k, v, a, **kw):
        return qu.masked_dilated_self_attention_1d(
            q, k, v, self.block_length, self.block_width, self.gap_size,
            self.num_memory_blocks)


class UnmaskedDilatedAttn(q.Attention):
    num_memory_blocks = 2
    block_length = 128
    block_width = 128
    gap_size = 0

    def attention(self, q, k, v, a, **kw):
        return qu.dilated_self_attention_1d(q, k, v, self.block_length,
                                            self.block_width, self.gap_size,
                                            self.num_memory_blocks)


_attentions = {
    'dot_product': q.DotProductAttn,
    'dot_product_relative': DotProductRelativeAttn,
    'dot_product_relative_v2': DotProductRelativeV2Attn,
    'dot_product_unmasked_relative_v2': DotProductUnmaskedRelativeAttn,
    'local_mask_right': LocalMaskRightAttn,
    'local_relative_mask_right': LocalRelativeMaskRightAttn,
    'local_unmasked': LocalUnmaskedAttn,
    'local_within_block_mask_right': LocalWithinBlockMaskRightAttn,
    'masked_dilated_1d': MaskedDilatedAttn,
    'unmasked_dilated_1d': UnmaskedDilatedAttn,
}


class ConvReluConv(q.FForward):
    def call(self, ins, **kw):
        PS = self.PS
        x = ins
        y = qu.preprocess(x, PS)
        y = qu.conv_relu_conv(
            y,
            PS.filter_size,
            PS.hidden_size,
            first_kernel_size=PS.conv_first_kernel,
            second_kernel_size=1,
            padding=self.conv_padding,
            dropout=PS.relu_drop,
        )
        return qu.postprocess(x, y, PS)


class ParamAttn(q.FForward):
    def call(self, ins, **kw):
        PS = self.PS
        x = ins
        y = qu.preprocess(x, PS)
        y = qu.parameter_attention(
            y,
            PS.param_attn_k_depth or PS.hidden_size,
            PS.param_attn_v_depth or PS.hidden_size,
            PS.hidden_size,
            PS.filter_size,
            PS.num_heads,
            PS.attn_drop,
        )
        return qu.postprocess(x, y, PS)


class ConvReluSep(q.FForward):
    def call(self, ins, **kw):
        PS = self.PS
        x = ins
        y = qu.preprocess(x, PS)
        y = qu.conv_hidden_relu(
            y,
            PS.filter_size,
            PS.hidden_size,
            kernel_size=(3, 1),
            second_kernel_size=(31, 1),
            padding='LEFT',
            dropout=PS.relu_drop,
        )
        return qu.postprocess(x, y, PS)


class SRU(q.FForward):
    def call(self, ins, **kw):
        PS = self.PS
        x = ins
        y = qu.preprocess(x, PS)
        y = qu.sru(y)
        return qu.postprocess(x, y, PS)


_fforwards = {
    'dense_relu_dense': q.DenseReluDense,
    'conv_relu_conv': ConvReluConv,
    'parameter_attention': ParamAttn,
    'conv_hidden_relu_with_sepconv': ConvReluSep,
    'sru': SRU,
}
