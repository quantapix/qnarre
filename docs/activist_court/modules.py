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

ks = tf.keras


class Encoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y = x
        y = self.reflect(y + [y[0]])
        y = self.conclude(y)
        return y


class Decoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.consider = Attention(layer, 'cnsd')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y, ye = x[:-1], x[-1]
        y = self.reflect(y + [y[0]])
        y = self.consider(y + [ye])
        y = self.conclude(y)
        return y


class Attention(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        self.layer = layer
        ps = layer.ps
        h = ps.dim_hidden
        self.num_heads = n = ps.num_heads or 1
        assert h % n == 0
        k = ps.dim_attn_k or ps.dim_attn or h
        assert k % n == 0
        self.scale = 1 / (k**0.5)
        v = ps.dim_attn_v or k
        assert v % n == 0
        self.drop_rate = ps.drop_attn or ps.drop_hidden
        with self.name_scope:
            self.q = layer.add_weight('q', shape=(h, n * k))
            self.k = layer.add_weight('k', shape=(h, n * k))
            self.v = layer.add_weight('v', shape=(h, n * v))
            self.y = layer.add_weight('y', shape=(n * v, h))

    @tf.function
    def __call__(self, x):
        inp, lens, ctx = x
        off = tf.math.reduce_max(lens)
        x = self.layer.pre_proc(inp[:, -off:, :])
        q = tf.einsum('bxi,ij->bxj', x, self.q)
        q = self.split_heads(q)
        k = tf.einsum('bci,ij->bcj', ctx, self.k)
        k = self.split_heads(k)
        v = tf.einsum('bci,ij->bcj', ctx, self.v)
        v = self.split_heads(v)
        y = tf.einsum('bnxi,bnci->bnxc', q, k)
        # use lens
        y = tf.nn.softmax(y * self.scale)
        y = self.layer.drop(y, self.drop_rate)
        y = tf.einsum('bnxc,bnci->bnxi', y, v)
        y = self.join_heads(y)
        y = tf.einsum('bxi,ij->bxj', y, self.y)
        y = self.layer.post_proc([x, y])
        y = tf.concat([inp[:, :-off, :], y], axis=1)
        return [y, lens]

    def split_heads(self, x):
        s = tf.shape(x)
        y = tf.reshape(x, [s[0], s[1], self.num_heads, -1])
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        return y

    @staticmethod
    def join_heads(x):
        y = tf.transpose(x, perm=[0, 2, 1, 3])
        s = tf.shape(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        return y


class Conclusion(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        self.layer = layer
        ps = layer.ps
        w = layer.width * ps.dim_hidden
        with self.name_scope:
            s = [w, ps.dim_dense]
            self.inflate = Dense(layer, 'infl', s, activation='relu')
            s = [ps.dim_dense, w]
            self.deflate = Dense(layer, 'defl', s, bias=False)

    @tf.function
    def __call__(self, x):
        y, lens = x
        w = self.layer.width
        d = self.layer.ps.dim_hidden
        y = tf.reshape(y, [-1, w * d])
        y = self.inflate(y)
        y = self.deflate(y)
        y = tf.reshape(y, [-1, w, d])
        return [y, lens]


class Dense(tf.Module):
    bias = None
    activation = None

    def __init__(self, layer, name, shape, activation=None, bias=True):
        super().__init__(name)
        with self.name_scope:
            self.kern = layer.add_weight('kern', shape=shape)
            if bias:
                self.bias = layer.add_weight('bias', shape=shape[1:])
            self.activation = ks.activations.get(activation)

    @tf.function
    def __call__(self, x):
        y = tf.einsum('bi,ij->bj', x, self.kern)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        if self.activation:
            y = self.activation(y)
        return y
