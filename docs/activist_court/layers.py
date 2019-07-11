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

import numpy as np
import tensorflow as tf

import data as qd
import modules as qm

ks = tf.keras


class Layer(ks.layers.Layer):
    def __init__(self, ps, **kw):
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        self.ps = ps


class ToRagged(ks.layers.Layer):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64)
    ] * 3])
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys


class Frames(Layer):
    @staticmethod
    def print_row(r):
        tf.print(
            tf.numpy_function(lambda ts: ''.join([qd.vocab[t] for t in ts]),
                              [r],
                              Tout=[tf.string]))
        return r

    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)
        s = (ps.dim_batch, ps.width_enc)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', shape=s, **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x
        ye = tf.concat([self.prev, xe], axis=1)
        el = tf.cast(xe.row_lengths(), dtype=tf.int32)
        ye = tf.gather_nd(ye, self.calc_idxs(el))
        ps = self.ps
        c = ps.width_dec - xd.bounding_shape(axis=1, out_type=tf.int32)
        yd = tf.pad(xd.to_tensor(), [[0, 0], [0, c]])
        dl = tf.cast(xd.row_lengths(), dtype=tf.int32)
        p = tf.concat([ye, xt], axis=1)
        tl = tf.cast(xt.row_lengths(), dtype=tf.int32)
        p = tf.gather_nd(p, self.calc_idxs(tl))
        self.prev.assign(p)
        if ps.print_frames:
            tf.print()
            tf.map_fn(self.print_row, self.prev)
        return [ye, el, yd, dl]

    def calc_idxs(self, lens):
        ps = self.ps
        b, w = ps.dim_batch, ps.width_enc
        y = tf.broadcast_to(tf.range(b)[:, None], [b, w])
        i = tf.range(w)[None, ] + lens[:, None]
        y = tf.stack([y, i], axis=2)
        return y


class Embed(Layer):
    @staticmethod
    def pos_timing(width, depth):
        assert depth % 2 == 0
        d = np.arange(depth)[np.newaxis, :]
        d = 1 / np.power(10000, (2 * (d // 2)) / np.float32(depth))
        t = np.arange(width)[:, np.newaxis] * d
        t = [np.sin(t[:, 0::2]), np.cos(t[:, 1::2])]
        t = np.concatenate(t, axis=-1)[np.newaxis, ...]
        t = tf.constant(t, dtype=tf.float32)
        return t

    def __init__(self, ps):
        super().__init__(ps)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight('emb', shape=s)
        self.enc_p = self.pos_timing(ps.width_enc, ps.dim_hidden)
        self.dec_p = self.pos_timing(ps.width_dec, ps.dim_hidden)

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        s = tf.shape(y)
        if s[-2] == self.ps.width_enc:
            y += tf.broadcast_to(self.enc_p, s)
        elif s[-2] == self.ps.width_dec:
            y += tf.broadcast_to(self.dec_p, s)
        else:
            pass
        y *= tf.cast(s[-1], tf.float32)**0.5
        return [y, lens]


class Encode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_enc
        n = ps.dim_stacks
        self.encs = [qm.Encoder(self, f'enc_{i}') for i in range(n)]

    @tf.function
    def call(self, x):
        y = x
        for e in self.encs:
            y = e(y)
        return y


class Decode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_dec
        n = ps.dim_stacks
        self.decs = [qm.Decoder(self, f'dec_{i}') for i in range(n)]

    @tf.function
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for d in self.decs:
            y = d(y + [ye])
        return y


class Debed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.dbd = qm.Dense(self, 'dbd', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.dbd(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y


class Probe(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.prb = qm.Dense(self, 'prb', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.prb(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y
