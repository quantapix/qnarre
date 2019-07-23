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

import datasets as qd
import modules as qm
import samples as qs
import utils as qu

ks = tf.keras
ki = ks.initializers


class Layer(ks.layers.Layer):
    initer = None

    @staticmethod
    def cfg_items(ps):
        yield from ps.cfg_items('initer_stddev', )

    @classmethod
    def from_config(cls, cfg):
        return cls(qu.Config(**cfg))

    def __init__(self, ps, **kw):
        kw.setdefault('name', qu.to_snake_case(type(self).__name__))
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        if isinstance(ps, qu.Config):
            self.cfg = ps
        else:
            self.cfg = qu.Config(**dict(self.cfg_items(ps)))
        cfg = self.cfg
        if cfg.initer_stddev:
            self.initer = ki.TruncatedNormal(stddev=cfg.initer_stddev)

    def get_config(self):
        b = super().get_config().items()
        c = self.cfg.items()
        return dict(list(b) + list(c))

    def add_weight(self, name, shape, **kw):
        kw.setdefault('initializer', self.initer)
        return super().add_weight(name=name, shape=shape, **kw)

    def drop(self, x, rate):
        y = x
        if self.cfg.runtime.is_training:
            y = tf.nn.dropout(x, rate)
        return y


class ToRagged(ks.layers.Layer):
    def __init__(self, **kw):
        kw.setdefault('name', qu.to_snake_case(type(self).__name__))
        super().__init__(**kw)

    @tf.function(
        input_signature=[[tf.TensorSpec(shape=[None], dtype=tf.int32)] * 8])
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
        s = [self.cfg.dim_batch, self.cfg.width_enc]
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', s, **kw)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_batch',
            'dim_hist',
            'width_dec',
            'width_enc',
        )

    def append(self, x, ragged):
        r, c = self.cfg.dim_batch, self.cfg.width_enc
        r = tf.broadcast_to(tf.range(r)[:, None], [r, c])
        lens = tf.cast(ragged.row_lengths(), dtype=tf.int32)[:, None]
        c = tf.range(c)[None, ] + lens
        y = tf.concat([x, ragged], axis=1)
        y = tf.gather_nd(y, tf.stack([r, c], axis=2))
        return [y, lens]

    def expand(self, x):
        c = self.cfg.width_dec - x.bounding_shape(axis=1, out_type=tf.int32)
        y = tf.pad(x.to_tensor(), [[0, 0], [0, c]])
        return y


class Tokens(Frames):
    def __init__(self, ps):
        super().__init__(ps)
        cfg = self.cfg
        assert cfg.width_enc > cfg.width_dec > 0
        s = [cfg.dim_batch, cfg.dim_hist]
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.hist = self.add_weight('hist', s, **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x[:3]
        ye, el = self.append(self.prev, xe)
        tf.debugging.assert_less_equal(el, self.cfg.width_enc)
        el = tf.concat([el, self.hist], axis=1)[:, :-1]
        yd = self.expand(xd)
        p, dl = self.append(ye, xt)
        self.prev.assign(p)
        tf.debugging.assert_less_equal(dl, self.cfg.width_dec)
        self.hist.assign(tf.concat([dl, el], axis=1)[:, :-1])
        if self.cfg.runtime.print_toks:
            qu.print_toks(self.prev, qd.vocab)
        return [ye, el, yd, dl]


class Metas(Frames):
    @tf.function
    def call(self, x):
        xe, xd = x[3:]
        ye, _ = self.append(self.prev, xe)
        yd = self.expand(xd)
        p, _ = self.append(ye, xd)
        self.prev.assign(p)
        if self.cfg.runtime.print_toks:
            qu.print_toks(self.prev, qd.metas)
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
        cfg = self.cfg
        self.norm = qm.Normalization(self, 'norm', [cfg.dim_hidden])
        self.toks = self.add_weight('toks', [cfg.dim_vocab, cfg.dim_hidden])
        self.meta = self.add_weight('meta', [cfg.dim_metas, cfg.dim_hidden])
        self.e_pos = self.pos_timing(cfg.width_enc, cfg.dim_hidden)
        self.d_pos = self.pos_timing(cfg.width_dec, cfg.dim_hidden)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'dim_hist',
            'dim_metas',
            'dim_vocab',
            'drop_hidden',
            'width_dec',
            'width_enc',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, hist, ym = x
        y = tf.one_hot(y, self.cfg.dim_vocab)
        y = tf.einsum('bsi,ih->bsh', y, self.toks)
        # y = tf.nn.embedding_lookup(self.toks, y)
        ym = tf.one_hot(ym, self.cfg.dim_metas)
        y += tf.einsum('bsi,ih->bsh', ym, self.meta)
        s = tf.shape(y)
        if s[-2] == self.cfg.width_enc:
            y += self.segment(self.e_pos, hist, s)
        elif s[-2] == self.cfg.width_dec:
            y += tf.broadcast_to(self.d_pos, s)
        else:
            pass
        y *= tf.cast(s[-1], tf.float32)**0.5
        y = self.drop(y, self.cfg.drop_hidden)
        y = self.norm(y)
        return [y, hist[:, 0]]

    def segment(self, pos, hist, shape):
        y = tf.broadcast_to(pos, shape)
        for i in tf.range(self.cfg.dim_hist, 0, -1):
            r = tf.RaggedTensor.from_tensor(y, hist[:, i - 1])
            y = tf.concat([y, r], axis=1)[:, -shape[-2]:, :]
        return y


class Encode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        cfg = self.cfg
        self.width = cfg.width_enc
        self.norm = qm.Normalization(self, 'norm', [cfg.dim_hidden])
        n = cfg.dim_stacks
        self.encs = [qm.Encoding(self, f'encode_{i}') for i in range(n)]

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'activ_concl',
            'dim_attn',
            'dim_attn_qk',
            'dim_attn_v',
            'dim_concl',
            'dim_hidden',
            'dim_stacks',
            'drop_attn',
            'drop_concl',
            'drop_hidden',
            'num_heads',
            'width_enc',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y = x
        for enc in self.encs:
            y = enc(y)
        return y


class Decode(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        cfg = self.cfg
        self.width = cfg.width_dec
        self.norm = qm.Normalization(self, 'norm', [cfg.dim_hidden])
        n = cfg.dim_stacks
        self.decs = [qm.Decoding(self, f'decode_{i}') for i in range(n)]

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'activ_concl',
            'dim_attn',
            'dim_attn_qk',
            'dim_attn_v',
            'dim_concl',
            'dim_hidden',
            'dim_stacks',
            'drop_attn',
            'drop_concl',
            'drop_hidden',
            'num_heads',
            'width_dec',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, None])
    ]])
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for dec in self.decs:
            y = dec(y + [ye])
        return y


class Debed(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        s = [self.cfg.dim_hidden, self.cfg.dim_vocab]
        self.inflate = qm.Dense(self, 'inflate', s)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'dim_vocab',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = self.inflate(y)
        y = tf.RaggedTensor.from_tensor(y, lens).to_tensor()
        return y


class Deduce(Layer):
    def __init__(self, ps, embed, decode):
        super().__init__(ps)
        self.embed = embed
        self.decode = decode
        s = [self.cfg.dim_hidden, self.cfg.dim_vocab]
        self.inflate = qm.Dense(self, 'inflate', s)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'dim_vocab',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, None])
    ]])
    def call(self, x):
        toks, *x = x
        if self.cfg.runtime.print_toks:
            qu.print_toks(toks, qd.vocab)
        y = self.deduce([toks] + x)
        n = tf.shape(y)[1]
        p = tf.shape(toks)[1] - n
        for i in tf.range(n):
            t = toks[:, :n]
            m = tf.equal(t, qd.MSK)
            if tf.equal(tf.reduce_any(m), True):
                t = self.update(t, m, y)
                if self.cfg.runtime.print_toks:
                    qu.print_toks(t, qd.vocab)
                toks = tf.pad(t, [[0, 0], [0, p]])
                y = self.deduce([toks] + x)
            else:
                e = tf.equal(t, qd.EOS)
                e = tf.math.count_nonzero(e, axis=1)
                if tf.equal(tf.reduce_any(tf.not_equal(e, 1)), False):
                    break
        return y

    def deduce(self, x):
        y = self.embed(x[:-1])
        y, lens = self.decode(y + x[-1:])
        y = self.inflate(y)
        y = tf.RaggedTensor.from_tensor(y, lens).to_tensor()
        return y

    def update(self, toks, msks, ctx):
        i = tf.cast(msks, tf.int32)
        i = tf.argmax(i, axis=1, output_type=tf.int32)
        n = tf.shape(msks)[0]
        i = tf.stack([tf.range(n), i], axis=1)
        m = tf.zeros_like(msks)
        m = tf.tensor_scatter_nd_update(m, i, tf.ones([n], tf.bool))
        y = tf.boolean_mask(ctx, m)
        y = tf.math.log_softmax(y)
        y = tf.argmax(y, axis=-1, output_type=tf.int32)
        y = tf.tensor_scatter_nd_update(toks, i, y)
        y = tf.where(tf.logical_and(msks, m), y, toks)
        return y


class Probe(Layer):
    def __init__(self, ps):
        super().__init__(ps)
        s = [self.cfg.dim_hidden, self.cfg.dim_vocab]
        self.inflate = qm.Dense(self, 'inflate', s)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'dim_vocab',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = self.inflate(y)
        y = tf.RaggedTensor.from_tensor(y, lens).to_tensor()
        return y


class Locate(Layer):
    span, spot = None, None

    def __init__(self, ps, group):
        super().__init__(ps)
        h = self.cfg.dim_hidden
        self.width = w = self.cfg.width_dec
        if group is qs.QAS:
            self.span = qm.Dense(self, 'span', [h * w, 2 * w])
        else:
            assert group is qs.FIX
            self.spot = qm.Dense(self, 'spot', [h * w, w])

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'width_dec',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, _ = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0], 1, -1])
        if self.span is not None:
            y = self.span(y)
            y = tf.reshape(y, [s[0], 2, -1])
        elif self.spot is not None:
            y = self.spot(y)
        return y
