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
# !pip install -U tf-nightly-2.0-preview

import numpy as np
import tensorflow as tf
import dataset as qd
import ragged as qr

ks = tf.keras
kl = ks.layers


def pos_timing(width, depth):
    assert depth % 2 == 0
    d = np.arange(depth)[np.newaxis, :]
    d = 1 / np.power(10000, (2 * (d // 2)) / np.float32(depth))
    t = np.arange(width)[:, np.newaxis] * d
    t = [np.sin(t[:, 0::2]), np.cos(t[:, 1::2])]
    t = np.concatenate(t, axis=-1)[np.newaxis, ...]
    return t


"""
pos = pos_timing(50, 512)

plt.pcolormesh(pos[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
"""


class Layer(kl.Layer):
    def __init__(self, ps, **kw):
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        self.ps = ps


class Embed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight(name='emb', shape=s)
        p = pos_timing(ps.len_max_input, ps.dim_hidden)
        p = tf.constant(p, dtype=tf.float32)
        self.pos = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb, x)
        y += tf.RaggedTensor.from_tensor(self.pos, lengths=y.row_lengths())
        return y


class Encode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        y = x
        for e in self.encs:
            y, ctx = e(y)
        return [y, ctx]


class Decode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        y, ctx = x
        for d in self.decs:
            y = d([y, ctx])
        return y


class Debed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.max_len = u = ps.len_max_input
        s = [u * ps.dim_hidden, ps.dim_vocab]
        self.dbd = Dense(self, 'dbd', s)

    def call(self, x):
        y = x.to_tensor()
        s = tf.shape(y)
        y = tf.pad(y, [[0, 0], [0, self.max_len - s[-2]], [0, 0]])
        y = tf.reshape(y, [-1, self.max_len * s[-1]])
        y = self.dbd(y)
        return y


class Encoder(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.conclude = Conclusion(layer, 'conc')

    @tf.Module.with_name_scope
    def __call__(self, x):
        y, ctx = self.reflect([x, None])
        y = self.conclude(y)
        return [y, ctx]


class Decoder(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.consider = Attention(layer, 'cnsd')
            self.conclude = Conclusion(layer, 'conc')

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, ctx = x
        y, _ = self.reflect([x, None])
        y, _ = self.consider([y, ctx])
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

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, ctx = x
        q = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.q))
        k = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.k))
        v = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.v))
        y = tf.einsum('bsi,bzi->bsz', q.to_tensor(), k.to_tensor())
        y = tf.nn.softmax(y * self.scale)
        y = tf.einsum('bsz,bzi->bsi', y, v.to_tensor())
        y = tf.RaggedTensor.from_tensor(y, lengths=x.row_lengths())
        return [y, tf.constant(1)]


class Conclusion(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        ps = layer.ps
        self.max_len = w = ps.len_max_input
        w *= ps.dim_hidden
        with self.name_scope:
            s = [w, ps.dim_dense]
            self.inflate = Dense(layer, 'infl', s, activation='relu')
            s = [ps.dim_dense, w]
            self.deflate = Dense(layer, 'defl', s, bias=False)

    @tf.Module.with_name_scope
    def __call__(self, x):
        y = x.to_tensor()
        s = tf.shape(y)
        y = tf.pad(y, [[0, 0], [0, self.max_len - s[-2]], [0, 0]])
        y = tf.reshape(y, [-1, self.max_len * s[-1]])
        y = self.inflate(y)
        y = self.deflate(y)
        y = tf.reshape(y, [-1, self.max_len, s[-1]])
        y = tf.RaggedTensor.from_tensor(y, lengths=x.row_lengths())
        return y


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

    @tf.Module.with_name_scope
    def __call__(self, x):
        y = tf.einsum('bi,ij->bj', x, self.kern)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        if self.activation:
            y = self.activation(y)
        return y


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = Embed(ps)(x)
    y = Encode(ps)(y)
    y = Decode(ps)(y)
    y = Debed(ps)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    len_max_input=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalAccuracy(),
    num_epochs=5,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)

if __name__ == '__main__':
    ps = qd.Params(**params)
    # import masking as qm
    # qm.main_graph(ps, qr.dset_for(ps), model_for(ps))
    qr.main_eager(ps, qr.dset_for(ps), model_for(ps))
