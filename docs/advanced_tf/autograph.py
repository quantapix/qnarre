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
import advanced_tf.dataset as qd
import advanced_tf.custom as qc

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
        self.enc_p = pos_timing(ps.width_enc, ps.dim_hidden)
        self.dec_p = pos_timing(ps.width_dec, ps.dim_hidden)

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        y *= y.shape[-1]**0.5
        w = tf.shape(y)
        if tf.equal(w[-2], self.ps.width_enc):
            y += tf.broadcast_to(self.enc_p, w)
        elif tf.equal(w[-2], self.ps.width_dec):
            y += tf.broadcast_to(self.dec_p, w)
        else:
            pass
        return [y, lens]


@tf.function
def print_prev(x):
    tf.print(''.join([qd.tokens[t] for t in x]))


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
    y = qc.Frames(ps)(y)
    embed = Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


if __name__ == '__main__':
    ps = qd.Params(**qc.params)
    import advanced_tf.masking as qm
    qm.main_graph(ps, qc.dset_for(ps), model_for(ps))
    # import advanced_tf.ragged as qr
    # qr.main_eager(ps, dset_for(ps), model_for(ps))
