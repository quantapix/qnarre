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
# !pip install tensorflow==2.0.0-beta0

import pathlib as pth
import tensorflow as tf

from datetime import datetime

td = tf.data
ks = tf.keras
kl = ks.layers

vocab = ('x', 'y')
vocab += ('+', '-', '*', '=', ',', ':')
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
def adapter(d, len_input):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1).to_tensor()
    x = tf.pad(x, [[0, 0], [0, len_input - tf.shape(x)[-1]]])
    y = tf.RaggedTensor.from_sparse(d['res']).to_tensor()
    return x, y[:, :1]


def dset_for(ps):
    ds = td.TFRecordDataset(list(paths(ps))).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(caster)
    return ds.map(lambda d: adapter(d, tf.constant(ps.len_input)))


class Layer(kl.Layer):
    def __init__(self, ps, **kw):
        kw.update(dtype=tf.float32)
        super().__init__(**kw)
        self.supports_masking = True
        self.ps = ps


class Masking(Layer):
    def __init__(self, *pa, **kw):
        super().__init__(*pa, **kw)
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, x, mask=None):
        return tf.not_equal(x, 0)

    def call(self, x):
        x._keras_mask = self.compute_mask(x)
        return x


class Embed(Layer):
    def build(self, shape):
        ps = self.ps
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb_t = self.add_weight(name='emb_t', shape=s)
        return super().build(shape)

    def compute_output_shape(self, shape):
        s = tf.TensorShape((self.ps.dim_hidden, ))
        return shape.concatenate(s)

    @tf.function
    def call(self, x, mask=None):
        y = tf.nn.embedding_lookup(self.emb_t, x)
        if mask is not None:
            y *= tf.cast(mask, tf.float32)[:, :, None]
        return y


class Reflect(Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q_w = self.add_weight(name='q_w', shape=(s, s))
        self.k_w = self.add_weight(name='k_w', shape=(s, s))
        self.v_w = self.add_weight(name='v_w', shape=(s, s))
        return super().build(shape)

    @tf.function
    def call(self, x, mask=None):
        q = tf.einsum('bsi,ij->bsj', x, self.q_w)
        k = tf.einsum('bsi,ij->bsj', x, self.k_w)
        y = tf.einsum('bsi,bzi->bsz', q, k) * self.scale
        if mask is not None:
            m = tf.logical_not(mask)
            m = tf.cast(m, tf.float32)[:, :, None]
            y += m * -1e9
        v = tf.einsum('bsi,ij->bsj', x, self.v_w)
        y = tf.einsum('bsz,bzi->bsi', tf.nn.softmax(y), v)
        return y


def model_for(ps):
    x = ks.Input(shape=(ps.len_input, ), dtype='int32')
    y = Masking(ps)(x)
    y = Embed(ps)(y)
    y = Reflect(ps)(y)
    y = kl.Reshape((ps.len_input * ps.dim_hidden, ))(y)
    y = kl.Dense(ps.dim_dense, activation='relu')(y)
    y = kl.Dense(ps.dim_vocab, name='out', activation=None)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metrics])
    print(m.summary())
    return m


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=15,
    dim_vocab=len(vocab) + 5,
    len_input=20,
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
    ps = Params(**params)
    ds = dset_for(ps)
    for s in ds.take(1):
        print(s)
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/qnarre/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=10)


if __name__ == '__main__':
    from absl import app
    app.run(main)
