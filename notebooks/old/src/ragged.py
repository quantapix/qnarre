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

import tensorflow as tf
import dataset as qd

ks = tf.keras
kl = ks.layers


@tf.function
def adapter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], qd.SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1)
    y = tf.RaggedTensor.from_sparse(d['res'])[:, :1].to_tensor()
    return (x.flat_values, x.row_splits), y


def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(adapter)


class Embed(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight(name='emb', shape=s)

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb, x)
        return y


class Reflect(kl.Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q = self.add_weight(name='q', shape=(s, s))
        self.k = self.add_weight(name='k', shape=(s, s))
        self.v = self.add_weight(name='v', shape=(s, s))
        return super().build(shape)

    def call(self, x):
        q = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.q))
        k = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.k))
        v = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.v))
        y = tf.einsum('bsi,bzi->bsz', q.to_tensor(), k.to_tensor())
        y = tf.nn.softmax(y * self.scale)
        y = tf.einsum('bsz,bzi->bsi', y, v.to_tensor())
        y = tf.RaggedTensor.from_tensor(y, lengths=x.row_lengths())
        return y


class Expand(kl.Layer):
    def __init__(self, ps):
        super().__init__()
        self.ps = ps

    def call(self, x):
        y = x.to_tensor()
        s = tf.shape(y)[-2]
        y = tf.pad(y, [[0, 0], [0, self.ps.len_max_input - s], [0, 0]])
        return y


def model_for(ps):
    x = [
        ks.Input(shape=(), dtype='int32'),  # , ragged=True)
        ks.Input(shape=(), dtype='int64'),
    ]
    y = Embed(ps)(x)
    y = Reflect()(y)
    y = Expand(ps)(y)
    y = kl.Reshape((ps.len_max_input * ps.dim_hidden, ))(y)
    y = kl.Dense(ps.dim_dense, activation='relu')(y)
    y = kl.Dense(ps.dim_vocab, name='dbd', activation=None)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


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
    # import advanced_tf.masking as qm
    # qm.main_graph(ps, dset_for(ps), model_for(ps))
    main_eager(ps, dset_for(ps), model_for(ps))
