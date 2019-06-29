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

from datetime import datetime

import advanced_tf.custom as qc
import advanced_tf.dataset as qd

ks = tf.keras
kl = ks.layers


def pos_timing(width, depth):
    assert depth % 2 == 0
    d = np.arange(depth)[np.newaxis, :]
    d = 1 / np.power(10000, (2 * (d // 2)) / np.float32(depth))
    t = np.arange(width)[:, np.newaxis] * d
    t = [np.sin(t[:, 0::2]), np.cos(t[:, 1::2])]
    t = np.concatenate(t, axis=-1)[np.newaxis, ...]
    t = tf.constant(t, dtype=tf.float32)
    return t


class Embed(qc.Embed):
    def __init__(self, ps):
        super().__init__(ps)
        p = pos_timing(ps.width_enc, ps.dim_hidden)
        self.enc_p = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])
        p = pos_timing(ps.width_dec, ps.dim_hidden)
        self.dec_p = tf.broadcast_to(p, [ps.dim_batch] + p.shape[1:])

    @tf.function
    def call(self, x):
        y, lens = x
        y += self.pos[:, :x.shape[1], :]
        return [y, lens]


tokens = qd.tokens
tokens.update({v: k for k, v in tokens.items()})


@tf.function
def print_prev(x):
    tf.print(''.join([tokens[t] for t in x]))


class Frames(qc.Frames):
    @tf.function
    def call(self, x):
        super().call(x)
        # tf.map_fn(print_prev, self.prev)
        return x


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = qc.ToRagged()(x)
    y = Frames(ps)(y)
    embed = Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    print(m.summary())
    return m


def main_graph(_):
    ps = qd.Params(**qc.params)
    ds = qc.dset_for(ps)
    m = model_for(ps)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


def main_eager(_):
    ps = qd.Params(**qc.params)
    ds = qc.dset_for(ps)
    m = model_for(ps)

    def step(x, y):
        with tf.GradientTape() as tape:
            yy = m(x)
            loss = ps.loss(y, yy)
            loss += sum(m.losses)
            xent = ps.metric(y, yy)
        for v in m.trainable_variables:
            print('---', v)
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
        print(f'Epoch {e} loss:', loss, ', xent:', xent)


if __name__ == '__main__':
    # tf.autograph.set_verbosity(10)
    from absl import app
    app.run(main_graph)
    # app.run(main_eager)
