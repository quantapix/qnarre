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

vocab = ('x', 'y', '+', '-', '*', '=', ',', ':')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

tokens = {k: v for v, k in enumerate(vocab, start=5)}
tokens.update({v: k for k, v in tokens.items()})

SEP = tokens[':']


def paths(ps):
    d = pth.Path('/tmp/qnarre/dataset')
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        yield str(d / f'shard_{i}.tfrecords')


@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}


@tf.function
def adapter(d, len_max_seq):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    s = tf.fill([ds.nrows(), 1], SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, s, os], axis=1)
    y = tf.RaggedTensor.from_sparse(d['res'])[:, :1].to_tensor()
    return (x.flat_values, x.row_splits), y


def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(paths(ps))).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(caster)
    return ds.map(lambda d: adapter(d, tf.constant(ps.len_max_seq)))


class Embed(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb_t = self.add_weight(name='emb_t', shape=s)
        p = pos_timing(ps.len_max_seq, ps.dim_hidden)
        p = tf.constant(p, dtype=tf.float32)
        self.pos_b = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb_t, x)
        # y *= y.shape[-1]**0.5
        y += tf.RaggedTensor.from_tensor(self.pos_b, lengths=y.row_lengths())
        return y


class Encode(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        self.ps = ps
        n = self.ps.dim_stacks
        self.encs = [Encoder(self, f'enc_{i}') for i in range(n)]

    def call(self, x):
        y = x
        for e in self.encs:
            y, ctx = e(y)
        return [y, ctx]


class Decode(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        self.ps = ps
        n = self.ps.dim_stacks
        self.decs = [Decoder(self, f'dec_{i}') for i in range(n)]

    def call(self, x):
        y, ctx = x
        for d in self.decs:
            y, _ = d([y, ctx])
        return y


class Debed(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        self.out = kl.Dense(ps.dim_vocab, activation=None)

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
        y, ctx = self.reflect([x, None])
        y = self.conclude(y)
        return y, ctx


class Decoder(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, name='_refl')
            self.consider = Attention(layer, name='_cons')
            self.conclude = Conclusion(layer, name=name + '_conc')

    @tf.Module.with_name_scope
    def __call__(self, x):
        x, ctx = x
        y, _ = self.reflect([x, None])
        y, _ = self.consider([y, ctx])
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
        return [y, None]


class Conclusion(tf.Module):
    def __init__(self, layer, name=None):
        super().__init__(name=name)
        ps = layer.ps
        self.max_len = m = ps.len_max_seq
        self.inflate = kl.Dense(ps.dim_dense, activation='relu')
        self.deflate = kl.Dense(m * ps.dim_hidden, use_bias=False)

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


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = Embed(ps)(x)
    y = Encode(ps)(y)
    y = Decode(ps)(y)
    y = Debed(ps)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metrics])
    print(m.summary())
    return m


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(vocab) + 5,
    len_max_seq=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=ks.metrics.SparseCategoricalAccuracy(),
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main(_):
    # tf.autograph.set_verbosity(1)
    ps = Params(**params)
    ds = dset_for(ps)
    # for s in ds.take(1):
    #     print(s)
    m = model_for(ps)
    from datetime import datetime
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/qnarre/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=10)


def main_eager(_):
    ps = Params(**params)
    ds = dset_for(ps)
    m = model_for(ps)

    def step(x, y):
        with tf.GradientTape() as tape:
            logits = m(x)
            loss = ps.loss(y, logits)
            loss += sum(m.losses)
            acc = ps.metrics(y, logits)
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
                m = ps.metrics.result()
                tf.print('Step:', s, ', loss:', loss, ', acc:', m)
        return loss, acc

    for e in range(10):
        loss, acc = epoch()
        print(f'Epoch {e} loss:', loss, ', acc:', acc)


if __name__ == '__main__':
    from absl import app  # , logging
    # logging.set_verbosity(logging.DEBUG)
    app.run(main_eager)
