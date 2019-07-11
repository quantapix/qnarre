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
# export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

from datetime import datetime

import numpy as np
import tensorflow as tf

import data as qd

ks = tf.keras
kl = ks.layers


class Layer(kl.Layer):
    def __init__(self, ps, **kw):
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        self.ps = ps


class ToRagged(kl.Layer):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64)
    ] * 3])
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys


class Frames(Layer):
    @staticmethod
    def print_row(r):
        tf.print(
            tf.numpy_function(lambda ts: ''.join([qd.vocab[t] for t in ts]),
                              [r],
                              Tout=[tf.string]))
        return r

    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)
        s = (ps.dim_batch, ps.width_enc)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', shape=s, **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x
        ye = tf.concat([self.prev, xe], axis=1)
        el = tf.cast(xe.row_lengths(), dtype=tf.int32)
        ye = tf.gather_nd(ye, self.calc_idxs(el))
        ps = self.ps
        c = ps.width_dec - xd.bounding_shape(axis=1, out_type=tf.int32)
        yd = tf.pad(xd.to_tensor(), [[0, 0], [0, c]])
        dl = tf.cast(xd.row_lengths(), dtype=tf.int32)
        p = tf.concat([ye, xt], axis=1)
        tl = tf.cast(xt.row_lengths(), dtype=tf.int32)
        p = tf.gather_nd(p, self.calc_idxs(tl))
        self.prev.assign(p)
        if ps.print_frames:
            tf.print()
            tf.map_fn(self.print_row, self.prev)
        return [ye, el, yd, dl]

    def calc_idxs(self, lens):
        ps = self.ps
        b, w = ps.dim_batch, ps.width_enc
        y = tf.broadcast_to(tf.range(b)[:, None], [b, w])
        i = tf.range(w)[None, ] + lens[:, None]
        y = tf.stack([y, i], axis=2)
        return y


class Embed(Layer):
    @staticmethod
    def pos_timing(width, depth):
        assert depth % 2 == 0
        d = np.arange(depth)[np.newaxis, :]
        d = 1 / np.power(10000, (2 * (d // 2)) / np.float32(depth))
        t = np.arange(width)[:, np.newaxis] * d
        t = [np.sin(t[:, 0::2]), np.cos(t[:, 1::2])]
        t = np.concatenate(t, axis=-1)[np.newaxis, ...]
        t = tf.constant(t, dtype=tf.float32)
        return t

    def __init__(self, ps):
        super().__init__(ps)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight('emb', shape=s)
        self.enc_p = self.pos_timing(ps.width_enc, ps.dim_hidden)
        self.dec_p = self.pos_timing(ps.width_dec, ps.dim_hidden)

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        s = tf.shape(y)
        if s[-2] == self.ps.width_enc:
            y += tf.broadcast_to(self.enc_p, s)
        elif s[-2] == self.ps.width_dec:
            y += tf.broadcast_to(self.dec_p, s)
        else:
            pass
        y *= tf.cast(s[-1], tf.float32)**0.5
        return [y, lens]


class Encode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_enc
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y = x
        for e in self.encs:
            y = e(y)
        return y


class Decode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_dec
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for d in self.decs:
            y = d(y + [ye])
        return y


class Debed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.dbd = Dense(self, 'dbd', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.dbd(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y


class Probe(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.prb = Dense(self, 'prb', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.prb(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y


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


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = ToRagged()(x)
    y = Frames(ps)(y)
    embed = Embed(ps)
    ye = Encode(ps)(embed(y[:2]))
    yd = Decode(ps)(embed(y[2:]) + [ye[0]])
    y = Probe(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


params = dict(
    dim_batch=5,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=5,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    print_frames=False,
    width_dec=15,
    width_enc=25,
)


def cross_entropy(tgt, y):
    tgt = tf.reshape(tf.cast(tgt, tf.int64), [-1])
    s = tf.shape(y)
    y = tf.reshape(y, [-1, s[-1]])
    y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt, logits=y)
    return tf.reshape(y, s[:-1])


class Loss(ks.losses.Loss):
    def __init__(self):
        super().__init__(name='loss')

    def call(self, tgt, y):
        return cross_entropy(tgt, y)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, tgt, y, sample_weight=None):
        vs = cross_entropy(tgt, y)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


params.update(
    loss=Loss(),
    metric=Metric(),
)


def main_graph(ps, ds, m):
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


def main_eager(ps, ds, m):
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = m(x)
            loss = ps.loss(y, logits)
            loss += sum(m.losses)
            xent = ps.metric(y, logits)
        grads = tape.gradient(loss, m.trainable_variables)
        ps.optimizer.apply_gradients(zip(grads, m.trainable_variables))
        return loss, xent

    @tf.function
    def epoch():
        s, loss, xent = 0, 0.0, 0.0
        for x, y in ds:
            s += 1
            loss, xent = step(x, y)
            if tf.equal(s % 10, 0):
                e = ps.metric.result()
                tf.print('Step:', s, ', loss:', loss, ', xent:', e)
        return loss, xent

    for e in range(ps.num_epochs):
        loss, xent = epoch()
        print(f'Epoch {e} loss:', loss.numpy(), ', xent:', xent.numpy())


if __name__ == '__main__':
    ps = qd.Params(**params)
    main_graph(ps, qd.dset_for(ps), model_for(ps))
    # main_eager(ps, qd.dset_for(ps), model_for(ps))
