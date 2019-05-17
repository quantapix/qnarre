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
from qnarre.neura import utils
from qnarre.neura.layers import base


class Attn(base.Layer):
    v_w = prox_b = None

    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'dim_attn',
                'dim_attn_k',
                'dim_attn_v',
                'dim_hidden',
                'drop_attn',
                'drop_hidden',
                'num_heads',
                'bias_prox',
            ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        self.pre = owner.pre
        self.post = owner.post
        self.pos_x_b = owner.pos_x_b
        self.pos_p_b = owner.pos_p_b

    def build(self, input_shape):
        x, ctx = input_shape
        h = x[2]
        cfg = self.cfg
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
        if cfg.pos_emb is None:
            e = x[1] + (x[1] if ctx is None else ctx[1])
            self.pos = utils.pos_timing(h, e)
            self.pos_w = self.add_weight('pos_w', (h, n * k))
            if self.pos_x_b is None:
                self.pos_x_b = self.add_bias('pos_x_b', (n, k))
            if self.pos_p_b is None:
                self.pos_p_b = self.add_bias('pos_p_b', (n, k))
        self.out_w = self.add_weight('out_w', (n * v, h))
        if cfg.bias_prox:
            self.prox_b = self.proximity(input_shape[1])
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    @tf.function
    def call(self, inputs, mask=None):
        x, ctx = inputs
        y = x if ctx is None else tf.concat([ctx, x], axis=1)
        if self.pre is not None:
            y = self.pre([y])
        if self.v_w is None:
            y = v = tf.einsum('bih,hk->bik', y, self.qkv_w)
        else:
            y = tf.einsum('bih,hk->bik', y, self.qk_w)
            v = tf.einsum('bih,hv->biv', v, self.v_w)
        xlen = tf.int_shape(x)[1]
        q = self.split_heads(y[:, -xlen:, :])
        k = self.split_heads(y)
        if self.pos is None:
            qk = tf.einsum('bnik,bnjk->bnij', q, k)
        else:
            qk = self.to_qk_with_pos(q, k)
        v = self.split_heads(v)
        y = self.to_scores(qk, mask[0], v)
        y = self.join_heads(y)
        y = tf.einsum('biv,vh->bih', y, self.out_w)
        if self.post is not None:
            y = self.post([x, y])
        return y

    def split_heads(self, x):
        s = tf.int_shape(x)
        n = self.cfg.num_heads
        y = tf.reshape(x, (-1, s[1], n, s[-1] // n))
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        return y

    def to_qk_with_pos(self, q, k):
        b = tf.expand_dims(tf.expand_dims(self.pos_x_b, axis=1), axis=3)
        y = tf.einsum('bnik,bnjk->bnij', q + b, k)
        p = tf.einsum('ih,hk->ik', self.pos, self.pos_w)
        p = tf.expand_dims(self.split_heads(p), axis=0)
        b = tf.expand_dims(tf.expand_dims(self.pos_b, axis=1), axis=3)
        p = tf.einsum('bnik,bnjk->bnij', q + b, p)
        y += self.shift(p)
        return y

    def shift(self, x):
        s = tf.int_shape(x)
        y = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        y = tf.reshape(y, [s[0], s[1], s[3] + 1, s[2]])
        y = tf.slice(y, [0, 0, 1, 0], [-1, -1, -1, -1])
        y = tf.reshape(y, s)
        return y

    def to_scores(self, qk, mask, v):
        b = tf.logical_not(mask)
        b = tf.cast(b, tf.floatx()) * utils.big_neg()
        if self.prox_b is not None:
            b += self.prox_b
        b = tf.expand_dims(tf.expand_dims(b, axis=1), axis=3)
        y = tf.softmax(qk * self.scale + b)
        cfg = self.cfg
        y = self.drop(y, cfg.drop_attn or cfg.drop_hidden)
        y = tf.einsum('bnij,bnjv->bniv', y, v)
        return y

    @staticmethod
    def join_heads(x):
        y = tf.transpose(x, perm=[0, 2, 1, 3])
        s = tf.int_shape(y)
        y = tf.reshape(y, (-1, s[1], s[2] * s[3]))
        return y

    @staticmethod
    def proximity(max_len):
        y = tf.range(max_len, dtype=tf.floatx())
        y = tf.expand_dims(y, axis=0) - tf.expand_dims(y, axis=1)
        y = -tf.log1p(tf.abs(y))
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        return y
