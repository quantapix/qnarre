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

from datetime import datetime

import tensorflow as tf
import dataset as qd

ks = tf.keras
kl = ks.layers


@tf.function
def adapter(d, len_max_input):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], qd.SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1).to_tensor()
    x = tf.pad(x, [[0, 0], [0, len_max_input - tf.shape(x)[-1]]])
    y = tf.RaggedTensor.from_sparse(d['res'])[:, :1].to_tensor()
    return x, y


def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(lambda d: adapter(d, tf.constant(ps.len_max_input)))


class Layer(kl.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True


class Masking(Layer):
    def __init__(self):
        super().__init__()
        # self._compute_output_and_mask_jointly = True

    def compute_mask(self, x, mask=None):
        return tf.not_equal(x, 0)

    def call(self, x):
        # x._keras_mask = self.compute_mask(x)
        return x


class Embed(Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight(name='emb', shape=s)

    def call(self, x, mask=None):
        y = tf.nn.embedding_lookup(self.emb, x)
        if mask is not None:
            y *= tf.cast(mask, tf.float32)[:, :, None]
        return y


class Reflect(Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q = self.add_weight(name='q', shape=(s, s))
        self.k = self.add_weight(name='k', shape=(s, s))
        self.v = self.add_weight(name='v', shape=(s, s))
        return super().build(shape)

    def call(self, x, mask=None):
        q = tf.einsum('bsi,ij->bsj', x, self.q)
        k = tf.einsum('bsi,ij->bsj', x, self.k)
        y = tf.einsum('bsi,bzi->bsz', q, k) * self.scale
        if mask is not None:
            # tf.print(' *** applying mask')
            m = tf.logical_not(mask)
            m = tf.cast(m, tf.float32)[:, :, None]
            y += m * -1e9
        v = tf.einsum('bsi,ij->bsj', x, self.v)
        y = tf.einsum('bsz,bzi->bsi', tf.nn.softmax(y), v)
        return y


def model_for(ps):
    x = ks.Input(shape=(ps.len_max_input, ), dtype='int32')
    y = Masking()(x)
    y = Embed(ps)(y)
    y = Reflect()(y)
    y = kl.Reshape((ps.len_max_input * ps.dim_hidden, ))(y)
    y = kl.Dense(ps.dim_dense, activation='relu')(y)
    y = kl.Dense(ps.dim_vocab, name='dbd', activation=None)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


def main_graph(ps, ds, m):
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=15,
    dim_vocab=len(qd.vocab),
    len_max_input=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=10,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)

if __name__ == '__main__':
    ps = qd.Params(**params)
    main_graph(ps, dset_for(ps), model_for(ps))
