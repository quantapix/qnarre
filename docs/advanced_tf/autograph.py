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
import pathlib as pth
import tensorflow as tf

from datetime import datetime

ks = tf.keras
kl = ks.layers

vocab = (' ', '$', ':', '|', ',')
vocab += ('x', 'y', '=', '+', '-', '*')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

tokens = {c: i for i, c in enumerate(vocab)}
tokens.update({v: k for k, v in tokens.items()})

PAD, MSK, SEP, EOS = [tokens[c] for c in vocab[:4]]


def paths(ps):
    d = pth.Path('/tmp/q/dataset')
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        yield str(d / f'shard_{i}.tfrecords')


@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}


@tf.function
def adapter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    n = ds.nrows()
    os = tf.RaggedTensor.from_sparse(d['op'])
    inp = tf.concat([ds, tf.fill([n, 1], SEP), os], axis=1)
    out = tf.RaggedTensor.from_row_lengths([PAD] * n, [1] * n)
    rs = tf.RaggedTensor.from_sparse(d['res'])
    tgt = tf.concat([rs, tf.fill([rs.nrows(), 1], EOS)], axis=1)
    # return [inp, out], tgt
    return ([
        inp.flat_values,
        inp.row_splits,
        out.flat_values,
        out.row_splits,
    ], [
        tgt.flat_values,
        tgt.row_splits,
    ])


def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(paths(ps))).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(caster)
    return ds.map(adapter)


def pos_timing(width, depth):
    assert depth % 2 == 0
    d = np.arange(depth)[np.newaxis, :]
    d = 1 / np.power(10000, (2 * (d // 2)) / np.float32(depth))
    t = np.arange(width)[:, np.newaxis] * d
    t = [np.sin(t[:, 0::2]), np.cos(t[:, 1::2])]
    t = np.concatenate(t, axis=-1)[np.newaxis, ...]
    return t


class Embed(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb_t = self.add_weight(name='emb_t', shape=s)
        p = pos_timing(ps.len_max_input, ps.dim_hidden)
        p = tf.constant(p, dtype=tf.float32)
        self.pos_b = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb_t, x)
        y *= y.shape[-1]**0.5
        y += tf.RaggedTensor.from_tensor(self.pos_b, lengths=y.row_lengths())
        return y


class Contextual(kl.Layer):
    def __init__(self, ps, len_ctx):
        super().__init__()
        self.ps = ps
        self.len_ctx = len_ctx

    def build(self, shape):
        s = (shape[0], self.len_ctx, shape[-1])
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.ctx = self.add_weight(name='ctx', shape=s, **kw)
        return super().build(shape)

    def ctx_add(x):
        pass

    def ctx_update(x):
        pass


class Encode(Contextual):
    def __init__(self, ps):
        super().__init__(ps, ps.len_enc_ctx)
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        y = self.ctx_add(x)
        for e in self.encs:
            y = e(y)
        self.ctx_update(y)
        return y


class Decode(Contextual):
    def __init__(self, ps):
        super().__init__(ps, ps.len_dec_ctx)
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        x, enc_ctx = x
        y = self.ctx_add(x)
        for d in self.decs:
            y = d(y + [enc_ctx])
        self.ctx_update(y)
        return y


class Debed(kl.Layer):
    def __init__(self, ps):
        super().__init__()
        self.max_len = u = ps.len_max_input
        s = [u * ps.dim_hidden, ps.dim_vocab]
        self.out = Dense(self, s, name='out')

    def call(self, x):
        y = x.to_tensor()
        s = tf.shape(y)
        y = tf.pad(y, [[0, 0], [0, self.max_len - s[-2]], [0, 0]])
        y = tf.reshape(y, [-1, self.max_len * s[-1]])
        y = self.out(y)
        return y


class Encoder(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, name='_refl')
            self.conclude = Conclusion(layer, name=name + '_concl')

    @tf.Module.with_name_scope
    def __call__(self, x):
        y = self.reflect(x + [None])
        y = self.conclude(y)
        return y


class Decoder(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, name='_refl')
            self.consider = Attention(layer, name='_cons')
            self.conclude = Conclusion(layer, name=name + '_conc')

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, enc_ctx = x[:-1], x[-1]
        y = self.reflect(x + [None])
        y = self.consider(y + [enc_ctx])
        y = self.conclude(y)
        return y


class Attention(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        h = layer.ps.dim_hidden
        self.scale = 1 / (h**0.5)
        with self.name_scope:
            self.q_w = layer.add_weight(name='q_w', shape=(h, h))
            self.k_w = layer.add_weight(name='k_w', shape=(h, h))
            self.v_w = layer.add_weight(name='v_w', shape=(h, h))

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, ctx = x
        q = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.q_w))
        k = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.k_w))
        v = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.v_w))
        y = tf.einsum('bsi,bzi->bsz', q.to_tensor(), k.to_tensor())
        y = tf.nn.softmax(y * self.scale)
        y = tf.einsum('bsz,bzi->bsi', y, v.to_tensor())
        y = tf.RaggedTensor.from_tensor(y, lengths=x.row_lengths())
        return [y, tf.constant(1)]


class Conclusion(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        ps = layer.ps
        self.max_len = u = ps.len_max_input
        u *= ps.dim_hidden
        with self.name_scope:
            s = [u, ps.dim_dense]
            self.inflate = Dense(layer, s, name='infl', activ='relu')
            s = [ps.dim_dense, u]
            self.deflate = Dense(layer, s, name='defl', bias=False)

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
    activ = None
    bias = None

    def __init__(self, layer, shape, name=None, activ=None, bias=True):
        super().__init__(name=name)
        with self.name_scope:
            self.kern = layer.add_weight('kern', shape=shape)
            self.activ = ks.activations.get(activ)
            if bias:
                kw = dict(shape=shape[1:], initializer='zeros')
                self.bias = layer.add_weight('bias', **kw)

    @tf.Module.with_name_scope
    def __call__(self, x):
        y = tf.einsum('bi,ij->bj', x, self.kern)
        if self.bias is not None:
            y += self.bias
        if self.activ:
            return self.activ(y)
        return y


def model_for(ps):
    emb = Embed(ps)
    xi = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    yi = emb(xi)
    y = Encode(ps)(yi)
    xo = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    yo = emb(xo)
    y = Decode(ps)([yo, y])
    y = Debed(ps)(y)
    m = ks.Model(inputs=xi + xo, outputs=[y])
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metrics])
    print(m.summary())
    return m


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(vocab) + 5,
    len_dec_ctx=20,
    len_enc_ctx=100,
    len_max_input=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=ks.metrics.SparseCategoricalAccuracy(),
    num_epochs=2,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main_eager(_):
    ps = Params(**params)
    ds = dset_for(ps)
    m = model_for(ps)

    def step(x, y):
        with tf.GradientTape() as tape:
            yy = m(x)
            loss = ps.loss(y, yy)
            loss += sum(m.losses)
            acc = ps.metrics(y, yy)
        grads = tape.gradient(loss, m.trainable_variables)
        ps.optimizer.apply_gradients(zip(grads, m.trainable_variables))
        return loss, acc

    @tf.function
    def epoch():
        s, loss, acc = 0, 0.0, 0.0
        for x, y in ds:
            s += 1
            loss, acc = step(x, y)
            if tf.equal(s % 10, 0):
                a = ps.metrics.result()
                tf.print('Step:', s, ', loss:', loss, ', acc:', a)
        return loss, acc

    for e in range(ps.num_epochs):
        loss, acc = epoch()
        print(f'Epoch {e} loss:', loss, ', acc:', acc)


def main_graph(_):
    ps = Params(**params)
    ds = dset_for(ps)
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


if __name__ == '__main__':
    from absl import app
    app.run(main_eager)
