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
kl = ks.layers
ki = ks.initializers


class Layer(kl.Layer):
    initer = None

    def __init__(self, ps, **kw):
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        self.ps = ps
        self.dropouts = {}
        self.norm = kl.LayerNormalization()
        if ps.initer_stddev:
            self.initer = ki.TruncatedNormal(stddev=ps.initer_stddev)

    def add_weight(self, name, shape, **kw):
        kw.setdefault('initializer', self.initer)
        return super().add_weight(name=name, shape=shape, **kw)

    def drop(self, x, rate):
        d = self.dropouts.setdefault(rate, kl.Dropout(rate))
        return d(x)


class ToRagged(kl.Layer):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64)
    ] * 3 + [
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ] * 2])
    def call(self, x):
        efv, ers, dfv, drs, tfv, trs, em, dm = x
        return [
            tf.RaggedTensor.from_row_splits(efv, ers),
            tf.RaggedTensor.from_row_splits(dfv, drs),
            tf.RaggedTensor.from_row_splits(tfv, trs),
            tf.RaggedTensor.from_row_splits(em, ers),
            tf.RaggedTensor.from_row_splits(dm, drs),
        ]


class Frames(Layer):
    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', [ps.dim_batch, ps.width_enc], **kw)

    def append(self, x, ragged):
        ps = self.ps
        r, c = ps.dim_batch, ps.width_enc
        r = tf.broadcast_to(tf.range(r)[:, None], [r, c])
        lens = tf.cast(ragged.row_lengths(), dtype=tf.int32)
        c = tf.range(c)[None, ] + lens[:, None]
        y = tf.concat([x, ragged], axis=1)
        y = tf.gather_nd(y, tf.stack([r, c], axis=2))
        return [y, lens]

    def expand(self, x):
        c = self.ps.width_dec - x.bounding_shape(axis=1, out_type=tf.int32)
        y = tf.pad(x.to_tensor(), [[0, 0], [0, c]])
        return y


class Tokens(Frames):
    @staticmethod
    def print_row(r):
        tf.print(
            tf.numpy_function(lambda ts: ''.join([qd.vocab[t] for t in ts]),
                              [r],
                              Tout=[tf.string]))
        return r

    def __init__(self, ps):
        super().__init__(ps)
        assert ps.width_enc > 0
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.lens = self.add_weight('lens', [ps.dim_batch, ps.dim_queue], **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x[:3]
        ye, el = self.append(self.prev, xe)
        el = self.queue(self.lens, el)
        yd = self.expand(xd)
        p, dl = self.append(ye, xt)
        self.prev.assign(p)
        self.lens.assign(self.queue(el, dl))
        if self.ps.print_frames:
            tf.print()
            tf.map_fn(self.print_row, self.prev)
        return [ye, el, yd, dl[:, None]]

    def queue(self, x, lens):
        tf.debugging.assert_less_equal(lens, self.ps.width_enc)
        y = tf.concat([lens[:, None], x], axis=1)
        # i = tf.constant(1)
        # while tf.math.reduce_sum(y[:, i:], axis=1) >= self.ps.width_enc:
        #    i += 1
        return y[:, :-1]


class Metas(Frames):
    @tf.function
    def call(self, x):
        xe, xd = x[3:]
        ye, _ = self.append(self.prev, xe)
        yd = self.expand(xd)
        p, _ = self.append(ye, xd)
        self.prev.assign(p)
        return [ye, yd]


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
        self.toks = self.add_weight('toks', [ps.dim_vocab, ps.dim_hidden])
        self.meta = self.add_weight('meta', [ps.dim_metas, ps.dim_hidden])
        self.e_pos = self.pos_timing(ps.width_enc, ps.dim_hidden)
        self.d_pos = self.pos_timing(ps.width_dec, ps.dim_hidden)

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens, ym = x
        y = tf.nn.embedding_lookup(self.toks, y)
        ym = tf.one_hot(ym, self.ps.dim_metas)
        y += tf.einsum('bsi,ih->bsh', ym, self.meta)
        s = tf.shape(y)
        if s[-2] == self.ps.width_enc:
            y += tf.broadcast_to(self.e_pos, s)
        elif s[-2] == self.ps.width_dec:
            y += tf.broadcast_to(self.d_pos, s)
        else:
            pass
        y *= tf.cast(s[-1], tf.float32)**0.5
        # y = self.drop(y, self.ps.drop_hidden)
        # y = self.norm(y)
        return [y, lens[:, 0]]


class Encode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_enc
        n = ps.dim_stacks
        self.encs = [qm.Encoder(self, f'encode_{i}') for i in range(n)]

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
        self.decs = [qm.Decoder(self, f'decode_{i}') for i in range(n)]

    @tf.function
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for d in self.decs:
            y = d(y + [ye])
        return y


class Debed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.inflate = qm.Dense(self, 'inflate', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        y = self.inflate(y)
        y = y[:, :tf.math.reduce_max(lens), :]
        return y


class Probe(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.inflate = qm.Dense(self, 'inflate', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        y = self.inflate(y)
        y = y[:, :tf.math.reduce_max(lens), :]
        return y
