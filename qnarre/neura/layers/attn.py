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


class Attn(tf.Layer):
    v_net = None

    def __init__(self, PS, owner=None, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = owner.pre if owner else None
        self.post = owner.post if owner else None
        self.src_bias = owner.src_bias if owner else None
        self.mem_bias = owner.mem_bias if owner else None

    def build(self, input_shape):
        src = input_shape[0]
        d = src[2]
        PS = self.PS
        assert d == PS.dim_hidden
        n = PS.num_heads
        assert d % n == 0
        kd = PS.dim_k or PS.dim_attn or d
        assert kd % n == 0
        self.scale = 1 / (kd**0.5)
        vd = PS.dim_v or kd
        assert vd % n == 0
        kw = dict(kernel_initializer=PS.initializer, use_bias=False)
        if kd == vd:
            self.qkv_net = tf.Dense(n * kd, name='qkv', **kw)
        else:
            self.qk_net = tf.Dense(n * kd, name='qk', **kw)
            self.v_net = tf.Dense(n * vd, name='v', **kw)
        self.drop = tf.Dropout(PS.drop_attn or PS.drop_hidden)
        self.o_net = tf.Dense(d, name='out', **kw)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, **kw):
        src, bias, mem, ctx = inputs + [None] * (4 - len(inputs))
        slen = tf.shape(src)[1]
        ctx = src if ctx is None else ctx
        clen = tf.shape(ctx)[1]
        if self.pre is not None:
            src = self.pre([src], **kw)
        y = [ctx, src] if mem is None else [mem, ctx, src]
        y = tf.concat(y, axis=1)
        if self.v_net is None:
            y = self.qkv_net(y, **kw)
            v = y[:, -clen - slen:-slen, :]
        else:
            y = self.qk_net(y, **kw)
            v = self.v_net(y[:, -clen - slen:-slen, :], **kw)
        q = self.split_heads(y[:, -slen:, :])
        k = self.split_heads(y[:, -clen - slen:-slen, :])
        if mem is None:
            y = tf.matmul(q, k, transpose_b=True)
        else:
            m = self.split_heads(y[:, :clen, :])
            y = tf.matmul(q + self.src_bias, k, transpose_b=True)
            m = tf.matmul(q + self.mem_bias, m, transpose_b=True)
            y = y + self.shift(m)
        v = self.split_heads(v)
        y = self.scores(y, bias, v, **kw)
        y = self.join_heads(y)
        y = self.o_net(y, **kw)
        if self.post is not None:
            y = self.post([src, y], **kw)
        return y

    def split_heads(self, x):
        s = tf.int_shape(x)
        n = self.PS.num_heads
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

    def scores(self, x, bias, v, **kw):
        y = x * self.scale
        if bias is not None:
            y = y + bias
        y = tf.softmax(y, **kw)
        y = self.drop(y, **kw)
        y = tf.matmul(y, v)
        return y
