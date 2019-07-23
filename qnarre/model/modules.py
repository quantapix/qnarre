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

import utils as qu

ks = tf.keras


class Dense(tf.Module):
    bias = None
    activate = None

    def __init__(self, layer, name, shape, bias=True, activation=None):
        super().__init__(name)
        with self.name_scope:
            self.kernel = layer.add_weight('kernel', shape=shape)
            if bias:
                self.bias = layer.add_weight('bias', shape=shape[1:])
            if activation:
                self.activate = qu.activation(activation)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None])])
    def __call__(self, x):
        y = tf.einsum('bni,ij->bnj', x, self.kernel)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        if self.activate:
            y = self.activate(y)
        return y


class Normalization(tf.Module):
    epsilon = 1e-3

    def __init__(self, layer, name, shape):
        super().__init__(name)
        kw = dict(shape=shape, dtype=tf.float32)
        with self.name_scope:
            self.gamma = layer.add_weight('gamma', initializer='ones', **kw)
            self.beta = layer.add_weight('beta', initializer='zeros', **kw)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None])])
    def __call__(self, x):
        mean, variance = tf.nn.moments(x, -1, keepdims=True)
        kw = dict(offset=self.beta,
                  scale=self.gamma,
                  variance_epsilon=self.epsilon)
        y = tf.nn.batch_normalization(x, mean, variance, **kw)
        return y


class Encoding(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        with self.name_scope:
            self.reflect = Attention(layer, 'reflect')
            self.conclude = Conclusion(layer, 'conclude')

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def __call__(self, x):
        y = x
        y = self.reflect(y + [y[0]])
        y = self.conclude(y)
        return y


class Decoding(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name)
        with self.name_scope:
            self.reflect = Attention(layer, 'reflect')
            self.consider = Attention(layer, 'consider')
            self.conclude = Conclusion(layer, 'conclude')

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, None])
    ]])
    def __call__(self, x):
        y, ye = x[:-1], x[-1]
        y = self.reflect(y + [y[0]])
        y = self.consider(y + [ye])
        y = self.conclude(y)
        return y


class Attention(tf.Module):
    out = None

    def __init__(self, layer, name):
        super().__init__(name)
        self.layer = layer
        cfg = layer.cfg
        h = cfg.dim_hidden
        k = cfg.dim_attn_qk or cfg.dim_attn or h
        self.scale = 1 / (k**0.5)
        self.num_heads = n = cfg.num_heads or 1
        v = cfg.dim_attn_v
        if not v:
            assert h % n == 0
            v = h // n
        self.drop_rate = cfg.drop_attn or cfg.drop_hidden
        with self.name_scope:
            self.q = Dense(layer, 'q', [h, n * k])
            self.k = Dense(layer, 'k', [h, n * k])
            self.v = Dense(layer, 'v', [h, n * v])
            if n * v != h:
                self.out = Dense(layer, 'out', [n * v, h])

    @tf.function
    def __call__(self, x):
        inp, lens, ctx = x
        off = tf.math.reduce_max(lens)
        x = inp[:, -off:, :]
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(ctx))
        v = self.split_heads(self.v(ctx))
        y = tf.einsum('bnxi,bnci->bnxc', q, k)
        y *= self.scale
        # use lens
        y = tf.nn.softmax(y)
        # y = self.layer.drop(y, self.drop_rate)
        y = tf.einsum('bnxc,bnci->bnxi', y, v)
        y = self.join_heads(y)
        if self.out is not None:
            y = self.out(y)
        y = self.layer.drop(y, self.drop_rate)
        y = self.layer.norm(x + y)
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
        cfg = layer.cfg
        self.drop_rate = cfg.drop_concl or cfg.drop_hidden
        a, c, h = cfg.activ_concl, cfg.dim_concl, cfg.dim_hidden
        with self.name_scope:
            self.inflate = Dense(layer, 'inflate', [h, c], activation=a)
            self.deflate = Dense(layer, 'deflate', [c, h])

    @tf.function
    def __call__(self, x):
        x, lens = x
        y = self.inflate(x)
        y = self.deflate(y)
        y = self.layer.drop(y, self.drop_rate)
        y = self.layer.norm(x + y)
        return [y, lens]
