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

# import numpy as np
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
def adapter(d, len_seq):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1).to_tensor()
    x = tf.pad(x, [[0, 0], [0, len_seq - tf.shape(x)[-1]]])
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
    return ds.map(lambda d: adapter(d, tf.constant(ps.len_seq)))


class Embed(kl.Layer):
    def __init__(self, ps, **kw):
        super().__init__(**kw)
        self._compute_output_and_mask_jointly = True
        self.supports_masking = True
        self.ps = ps

    def build(self, shape):
        ps = self.ps
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb_t = self.add_weight(name='emb_t', shape=s, dtype=tf.float32)
        return super().build(shape)

    def compute_output_shape(self, shape):
        s = tf.TensorShape((self.ps.dim_hidden, ))
        return shape.concatenate(s)

    def compute_mask(self, x, mask=None):
        return tf.not_equal(x, 0)

    @tf.function
    def call(self, x):
        y = tf.nn.embedding_lookup(self.emb_t, x)
        y._keras_mask = self.compute_mask(x)
        return y


class Attn(kl.Layer):
    def __init__(self, ps, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.ps = ps

    def build(self, shape):
        s = shape[-1]
        kw = dict(dtype=tf.float32)
        self.qkv_w = self.add_weight(name='qkv_w', shape=(s, s), **kw)
        return super().build(shape)

    def compute_output_shape(self, shape):
        s = tf.TensorShape((self.ps.dim_hidden, ))
        return shape.concatenate(s)

    @tf.function
    def call(self, x, mask=None):
        if mask is not None:
            print('mask', mask.shape.as_list())
            m = tf.cast(tf.logical_not(mask), tf.float32) * -1e9
            x += m[:, :, None]
            print('xx', x.shape.as_list())
        y = tf.einsum('bi,ij->bj', x, self.dense_w) + self.dense_b
        return y


class Norm(kl.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True

    def build(self, shape):
        s = shape[-1]
        self.norm_w = self.add_weight('norm_w', s, initializer='ones')
        self.norm_b = self.add_weight('norm_b', s, initializer='zeros')
        return super().build(shape)

    @tf.function
    def call(self, x, mask=None):
        if mask is not None:
            x += tf.cast(mask, tf.float32)[:, :, None]
        m = tf.reduce_mean(x, axis=-1, keepdims=True)
        v = tf.reduce_mean(tf.square(x - m), axis=-1, keepdims=True)
        y = (x - m) / tf.sqrt(v + 1e-6)
        y = y * self.norm_w + self.norm_b
        return y


def model_for(ps):
    x = ks.Input(shape=(ps.len_seq, ), dtype='int32')
    y = Embed(ps)(x)
    print('y', y.shape.as_list())
    y = Dense(ps)(y)
    # y = Norm()(y)
    print('y', y.shape.as_list())
    y = kl.Dense(ps.dim_vocab, name='out', activation=None)(y)
    print('y', y.shape.as_list())
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metrics])
    print(m.summary())
    return m


params = dict(
    dim_batch=2,
    dim_hidden=10,
    dim_vocab=len(vocab) + 5,
    len_seq=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=ks.metrics.SparseCategoricalAccuracy(),
    num_layers=10,
    num_shards=10,
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
