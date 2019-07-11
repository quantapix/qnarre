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
        super().__init__(name=name)
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
        super().__init__(name=name)
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
        super().__init__(name=name)
        h = layer.ps.dim_hidden
        self.scale = 1 / (h**0.5)
        with self.name_scope:
            self.q = layer.add_weight('q', shape=(h, h))
            self.k = layer.add_weight('k', shape=(h, h))
            self.v = layer.add_weight('v', shape=(h, h))

    @tf.function
    def __call__(self, x):
        x, lens, ctx = x
        off = tf.math.reduce_max(lens)
        q = tf.einsum('bni,ij->bnj', x[:, -off:, :], self.q)
        k = tf.einsum('bni,ij->bnj', ctx, self.k)
        y = tf.einsum('bni,bmi->bnm', q, k)
        # use lens
        y = tf.nn.softmax(y * self.scale)
        v = tf.einsum('bni,ij->bnj', ctx, self.v)
        y = tf.einsum('bnm,bmi->bni', y, v)
        y = tf.concat([x[:, :-off, :], y], axis=1)
        return [y, lens]


class Conclusion(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
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
        super().__init__(name=name)
        with self.name_scope:
            kw = dict(shape=shape, initializer='glorot_uniform')
            self.kern = layer.add_weight('kern', **kw)
            if bias:
                kw.update(shape=shape[1:], initializer='zeros')
                self.bias = layer.add_weight('bias', **kw)
            self.activation = ks.activations.get(activation)

    @tf.function
    def __call__(self, x):
        y = tf.einsum('bi,ij->bj', x, self.kern)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        if self.activation:
            y = self.activation(y)
        return y
