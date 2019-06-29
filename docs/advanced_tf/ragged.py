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

from datetime import datetime

import advanced_tf.dataset as qd

ks = tf.keras
kl = ks.layers

SEP = qd.tokens[':']


@tf.function
def adapter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], SEP)
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
        self.emb_t = self.add_weight(name='emb_t', shape=s)

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb_t, x)
        return y


class Reflect(kl.Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q_w = self.add_weight(name='q_w', shape=(s, s))
        self.k_w = self.add_weight(name='k_w', shape=(s, s))
        self.v_w = self.add_weight(name='v_w', shape=(s, s))
        return super().build(shape)

    def call(self, x):
        q = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.q_w))
        k = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.k_w))
        v = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.v_w))
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
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    # , ragged=True)
    y = Embed(ps)(x)
    y = Reflect()(y)
    y = Expand(ps)(y)
    y = kl.Reshape((ps.len_max_input * ps.dim_hidden, ))(y)
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
    dim_vocab=len(qd.vocab),
    len_max_input=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=ks.metrics.SparseCategoricalAccuracy(),
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)


def main(_):
    ps = qd.Params(**params)
    ds = dset_for(ps)
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=10)


def main_eager(_):
    ps = qd.Params(**params)
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
    from absl import app
    app.run(main)
    # app.run(main_eager)
