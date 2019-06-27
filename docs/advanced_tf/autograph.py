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
    out = tf.RaggedTensor.from_tensor(tf.fill([n, 1], PAD))
    rs = tf.RaggedTensor.from_sparse(d['res'])
    tgt = tf.concat([rs, tf.fill([rs.nrows(), 1], EOS)], axis=1)
    # return [inp, out], tgt
    return ((inp.flat_values, inp.row_splits, out.flat_values, out.row_splits),
            tgt.to_tensor())


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
        w = max(ps.width_dec, ps.width_enc)
        p = pos_timing(w, ps.dim_hidden)
        p = tf.constant(p, dtype=tf.float32)
        self.pos_b = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb_t, x)
        y *= y.shape[-1]**0.5
        lens = tf.cast(y.row_lengths(), dtype=tf.int32)
        y += tf.RaggedTensor.from_tensor(self.pos_b, lengths=lens)
        return [y.to_tensor(), lens]


class Context(kl.Layer):
    def __init__(self, ps, width):
        super().__init__()
        self.ps = ps
        self.width = width
        s = (ps.dim_batch, self.width, ps.dim_hidden)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.ctx = self.add_weight(name='ctx', shape=s, **kw)

    def append(self, x):
        x, lens = x
        y = tf.concat([self.ctx, x], axis=1)
        y = tf.gather_nd(y, self.calc_idxs(lens))
        return [y, lens]

    def calc_idxs(self, lens):
        w = self.width
        tf.debugging.assert_greater_equal(w, lens)
        y = self.ps.dim_batch
        y = tf.broadcast_to(tf.range(y)[:, None], [y, w])
        i = tf.range(w)[None, ] + lens[:, None]
        y = tf.stack([y, i], axis=2)
        return y


class Encode(Context):
    def __init__(self, ps):
        super().__init__(ps, ps.width_enc)
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        y = self.append(x)
        for e in self.encs:
            y = e(y)
        y = y[0]
        self.ctx.assign(y)
        return y


class Decode(Context):
    def __init__(self, ps):
        super().__init__(ps, ps.width_dec)
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    def call(self, x):
        x, ye = x[:-1], x[-1]
        y = self.append(x)
        for d in self.decs:
            y = d(y + [ye])
        y = y[0]
        self.ctx.assign(y)
        return y


class Debed(kl.Layer):
    def __init__(self, ps):
        super().__init__()
        self.out = Dense(self, [ps.dim_hidden, ps.dim_vocab], name='out')

    def call(self, x):
        s = tf.shape(x)
        y = tf.reshape(x, [s[0] * s[1], -1])
        y = self.out(y)
        y = tf.reshape(y, [s[0], s[1], -1])
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
        x, ye = x[:-1], x[-1]
        y = self.reflect(x + [None])
        y = self.consider(y + [ye])
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
        x, lens, ctx = x
        off = tf.math.reduce_max(lens)
        q = tf.einsum('bni,ij->bnj', x[:, -off:, :], self.q_w)
        ctx = x if ctx is None else ctx
        k = tf.einsum('bni,ij->bnj', ctx, self.k_w)
        y = tf.einsum('bni,bmi->bnm', q, k)
        # use lens
        y = tf.nn.softmax(y * self.scale)
        v = tf.einsum('bni,ij->bnj', ctx, self.v_w)
        y = tf.einsum('bnm,bmi->bni', y, v)
        y = tf.concat([x[:, :-off, :], y], axis=1)
        return [y, lens]


class Conclusion(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        self.layer = layer
        ps = layer.ps
        w = layer.width * ps.dim_hidden
        with self.name_scope:
            s = [w, ps.dim_dense]
            self.inflate = Dense(layer, s, name='infl', activation='relu')
            s = [ps.dim_dense, w]
            self.deflate = Dense(layer, s, name='defl', bias=False)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, lens = x
        w = self.layer.width
        d = self.layer.ps.dim_hidden
        y = tf.reshape(x, [-1, w * d])
        y = self.inflate(y)
        y = self.deflate(y)
        y = tf.reshape(y, [-1, w, d])
        return [y, lens]


class Dense(tf.Module):
    activation = None
    bias = None

    def __init__(self, layer, shape, name=None, activation=None, bias=True):
        super().__init__(name=name)
        with self.name_scope:
            self.kern = layer.add_weight('kern', shape=shape)
            self.activation = ks.activations.get(activation)
            if bias:
                kw = dict(shape=shape[1:], initializer='zeros')
                self.bias = layer.add_weight('bias', **kw)

    @tf.Module.with_name_scope
    def __call__(self, x):
        y = tf.einsum('bi,ij->bj', x, self.kern)
        if self.bias is not None:
            y += self.bias
        if self.activation:
            return self.activation(y)
        return y


def model_for(ps):
    embed = Embed(ps)
    xe = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    ye = Encode(ps)(embed(xe))
    xd = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    yd = Decode(ps)(embed(xd) + [ye])
    y = Debed(ps)(yd)
    m = ks.Model(inputs=xe + xd, outputs=y)
    # m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


class Loss(ks.losses.Loss):
    @staticmethod
    def xent(y_true, y_pred):
        s = tf.shape(y_true)
        kw = dict(labels=y_true, logits=y_pred[:, :s[1], :])
        return tf.nn.sparse_softmax_cross_entropy_with_logits(**kw)

    def __init__(self):
        super().__init__(name='loss')

    def call(self, y_true, y_pred):
        return self.xent(y_true, y_pred)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        vs = Loss.xent(y_true, y_pred)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(vocab) + 5,
    loss=Loss(),
    metric=Metric(),
    num_epochs=2,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    width_dec=20,
    width_enc=100,
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
            acc = ps.metric(y, yy)
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
                a = ps.metric.result()
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
