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

from qnarre.neura import tf
from qnarre.neura.layers import base


class Attn(base.Layer):
    pre = post = src_b = mem_b = v_w = None

    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'dim_attn',
                'dim_attn_k',
                'dim_attn_v',
                'dim_hidden',
                'drop_attn',
                'drop_hidden',
                'num_heads',
            ))

    def __init__(self, params, owner=None, **kw):
        super().__init__(params, **kw)
        if owner:
            self.pre = owner.pre
            self.post = owner.post
            self.src_b = owner.src_b
            self.mem_b = owner.mem_b

    def build(self, input_shape):
        cfg = self.cfg
        src = input_shape[0]
        h = src[2]
        assert h == cfg.dim_hidden
        n = cfg.num_heads
        assert h % n == 0
        k = cfg.dim_attn_k or cfg.dim_attn or h
        assert k % n == 0
        self.scale = 1 / (k**0.5)
        v = cfg.dim_attn_v or k
        assert v % n == 0
        if k == v:
            self.qkv_w = self.add_weight('qkv_w', (h, n * k))
        else:
            self.qk_w = self.add_weight('qk_w', (h, n * k))
            self.v_w = self.add_weight('v_w', (h, n * v))
        self.out_w = self.add_weight('out_w', (n * v, h))
        if len(input_shape) > 2 and input_shape[2]:
            if self.src_b is None:
                self.src_b = self.add_weight('src_b', (n, k))
            if self.mem_b is None:
                self.mem_b = self.add_weight('mem_b', (n, k))
        return super().build(input_shape)

    @tf.function
    def call(self, inputs):
        src, bias, mem, ctx = inputs + [None] * (4 - len(inputs))
        slen = tf.shape(src)[1]
        ctx = src if ctx is None else ctx
        clen = tf.shape(ctx)[1]
        y = [ctx, src] if mem is None else [mem, ctx, src]
        y = tf.concat(y, axis=1)
        if self.pre is not None:
            y = self.pre([y])
        if self.v_w is None:
            y = tf.einsum('bih,hk->bik', y, self.qkv_w)
            v = y[:, -clen - slen:-slen, :]
        else:
            y = tf.einsum('bih,hk->bik', y, self.qk_w)
            v = y[:, -clen - slen:-slen, :]
            v = tf.einsum('bih,hv->biv', v, self.v_w)
        q = self.split_heads(y[:, -slen:, :])
        k = self.split_heads(y[:, -clen - slen:-slen, :])
        if mem is None:
            y = tf.einsum('bnik,bnjk->bnij', q, k)
        else:
            m = self.split_heads(y[:, :clen, :])
            b = tf.expand_dims(tf.expand_dims(self.src_b, axis=1), axis=3)
            y = tf.einsum('bnik,bnjk->bnij', q + b, k)
            b = tf.expand_dims(tf.expand_dims(self.mem_b, axis=1), axis=3)
            m = tf.einsum('bnik,bnjk->bnij', q + b, m)
            y = y + self.shift(m)
        v = self.split_heads(v)
        y = self.scores(y, bias, v)
        y = self.join_heads(y)
        y = tf.einsum('biv,vh->bih', y, self.out_w)
        if self.post is not None:
            y = self.post([src, y])
        return y

    def split_heads(self, x):
        s = tf.int_shape(x)
        n = self.cfg.num_heads
        y = tf.reshape(x, (-1, s[1], n, s[-1] // n))
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        return y

    @staticmethod
    def join_heads(x):
        y = tf.transpose(x, perm=[0, 2, 1, 3])
        s = tf.int_shape(y)
        y = tf.reshape(y, (-1, s[1], s[2] * s[3]))
        return y

    def shift(self, x):
        s = tf.shape(x)
        y = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        y = tf.reshape(y, [s[0], s[1], s[3] + 1, s[2]])
        y = tf.slice(y, [0, 0, 1, 0], [-1, -1, -1, -1])
        y = tf.reshape(y, s)
        return y

    def scores(self, x, bias, v):
        y = x * self.scale
        if bias is not None:
            y = y + bias
        y = tf.softmax(y)
        r = self.cfg.drop_attn or self.cfg.drop_hidden
        y = self.dropout(y, r)
        y = tf.einsum('bnij,bnjv->bniv', y, v)
        return y
