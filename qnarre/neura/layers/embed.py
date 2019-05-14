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

from qnarre.neura import tf
from qnarre.neura.layers import base


class TokEmbed(base.Layer):
    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'brackets',
                'dim_embed',
                'dim_hidden',
                'emb_one_hot',
                'num_toks',
            ))

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self.tbl_ws = []
        self.out_ws = []

    def build(self, input_shape):
        cfg = self.cfg
        h = cfg.dim_hidden
        d = cfg.dim_embed or h
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            t = d // (len(bs)**i)
            self.tbl_ws.append(self.add_weight(f'tbl_w{i}', (e - b, t)))
            o = self.add_weight(f'out_w{i}', (t, h)) if t != h else None
            self.out_ws.append(o)
            b = e
        self.one_hot = cfg.emb_one_hot
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.cfg.dim_hidden, )

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    @tf.function
    def call(self, inputs, **_):
        cfg = self.cfg
        x = inputs
        y = tf.zeros(tf.int_shape(x) + (cfg.dim_hidde, ))
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            m = (x >= (b or 1)) & (x < e)
            u = self.lookup(tf.boolean_mask(x, m) - b, i)
            y += tf.tensor_scatter_nd_update(y, tf.where(m), u)
        y *= tf.shape(y)[-1]**0.5
        return y

    def lookup(self, x, i):
        t = self.tbl_ws[i]
        if self.one_hot:
            y = tf.one_hot(x, tf.shape(t)[0], axis=-1)
            y = tf.einsum('ne,bin->bie', t, y)
        else:
            y = tf.embedding_lookup(t, x)
        o = self.out_ws[i]
        if o is not None:
            y = tf.einsum('bie,eh->bih', y, o)
        return y


class TypEmbed(base.Layer):
    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'tok_types',
            ))

    def build(self, input_shape):
        cfg = self.cfg
        src, typ = input_shape
        _, tlen, h = src
        assert tlen == typ[1]
        self.typ_w = self.add_weight('typ_w', (cfg.tok_types, h))
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        src, typ = inputs
        y = typ * tf.cast(mask[0], typ.dtype)
        y = tf.one_hot(y, self.cfg.tok_types)
        return src + tf.einsum('bihj,h->bih', y, self.typ_w)


class PosEmbed(base.Layer):
    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'pos_max',
                'ctx_len',
                'tgt_len',
            ))

    def build(self, input_shape):
        cfg = self.cfg
        _, tlen, h = input_shape
        plen = max(cfg.pos_max or 0, cfg.ctx_len, cfg.tgt_len)
        assert tlen <= plen
        self.pos_b = self.add_weight('pos_b', (plen, h))[:tlen, :]
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        y = tf.cast(mask, self.pos_b.dtype)
        y = tf.einsum('bihj,h->bih', self.pos_b, y)
        return inputs + y


class PosTiming(base.Layer):
    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'start',  # 0
                'min_scale',  # 1.0
                'max_scale',  # 1.0e4
            ))

    def build(self, input_shape):
        cfg = self.cfg
        _, tlen, h = input_shape
        assert h % 2 == 0
        n = h // 2
        s = np.log(cfg.max_scale / cfg.min_scale) / max(n - 1, 1)
        s = cfg.min_scale * tf.exp(tf.range(float(n)) * -s)
        p = tf.range(float(tlen)) + cfg.start
        p = tf.expand_dims(p, axis=1) * tf.expand_dims(s, axis=0)
        self.pos_b = tf.concat([tf.sin(p), tf.cos(p)], axis=1)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        y = tf.cast(mask, self.pos_b.dtype)
        y = tf.einsum('bihj,h->bih', self.pos_b, y)
        return inputs + y
