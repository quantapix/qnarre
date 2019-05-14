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
    def __init__(self, params, **kw):
        super().__init__(params, **kw)

    def build(self, input_shape):
        cfg = self.cfg
        tok, typ = input_shape
        _, tlen, hsize = tok
        assert tlen == typ[1]
        sh = (cfg.token_types, hsize)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        tok, typ = inputs
        y = typ * tf.cast(mask[0], typ.dtype)
        y = tf.one_hot(y, self.cfg.token_types)
        return tok + tf.matmul(y, self.gain)


class PosEmbed(base.Layer):
    def __init__(self, params, **kw):
        super().__init__(params, **kw)

    def build(self, input_shape):
        cfg = self.cfg
        _, tlen, hsize = input_shape
        plen = max(cfg.max_pos or 0, cfg.ctx_len, cfg.tgt_len)
        assert tlen <= plen
        sh = (plen, hsize)
        b = self.add_weight(shape=sh, initializer=cfg.initializer)
        b = b[:tlen, :]
        self.bias = tf.expand_dims(b, axis=0)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        y = tf.cast(mask, self.bias.dtype)
        y = self.bias * tf.expand_dims(y, axis=2)
        return inputs + y


class PosTiming(base.Layer):
    start = 0
    min_scale = 1.0
    max_scale = 1.0e4

    def __init__(self, params, start=None, min_scale=None, max_scale=None, **kw):
        super().__init__(params, **kw)
        if start:
            self.start = start
        if min_scale:
            self.min_scale = float(min_scale)
        if max_scale:
            self.max_scale = float(max_scale)

    def build(self, input_shape):
        _, tlen, hsize = input_shape
        assert hsize % 2 == 0
        n = hsize // 2
        s = np.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * tf.exp(tf.range(n, dtype=tf.floatx()) * -s)
        p = tf.range(tlen, dtype=tf.floatx()) + self.start
        p = tf.expand_dims(p, axis=1) * tf.expand_dims(s, axis=0)
        p = tf.concat([tf.sin(p), tf.cos(p)], axis=1)
        self.bias = tf.expand_dims(p, axis=0)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask):
        y = tf.cast(mask, self.bias.dtype)
        y = self.bias * tf.expand_dims(y, axis=2)
        return inputs + y
