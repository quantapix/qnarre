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

from qnarre.neura import tf
from qnarre.neura import utils

from qnarre.neura.layers.base import Layer


class TokEmbed(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'PAD',
                'brackets',
                'dim_embed',
                'dim_hidden',
                'emb_one_hot',
                'num_toks',
            ))

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        self._compute_output_and_mask_jointly = True
        self.table_ws = []
        self.adapt_ws = []

    def build(self, input_shape):
        cfg = self.cfg
        h = cfg.dim_hidden
        d = cfg.dim_embed or h
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        assert b == cfg.PAD
        for i, e in enumerate(bs):
            p = d // (len(bs)**i)
            t = self.add_weight(f'table_w{i}', (e - b, p))
            self.table_ws.append(t)
            a = None if p == h else self.add_weight(f'adapt_w{i}', (p, h))
            self.adapt_ws.append(a)
            b = e
        self.one_hot = cfg.emb_one_hot
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        s = tf.TensorShape((self.cfg.dim_hidden, ))
        return input_shape.concatenate(s)

    @tf.function
    def call(self, inputs):
        x = inputs
        cfg = self.cfg
        y = tf.zeros(tf.int_shape(x) + (cfg.dim_hidden, ))
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            m = (x >= (b or 1)) & (x < e)
            u = tf.boolean_mask(x, m)
            u = self.lookup(u - b, i)
            y = tf.tensor_scatter_nd_add(y, tf.where(m), u)
            b = e
        y *= y.shape[-1]**0.5
        y._keras_mask = tf.not_equal(x, cfg.PAD)
        return y

    def lookup(self, x, i):
        t = self.table_ws[i]
        if self.one_hot:
            y = tf.one_hot(x, tf.shape(t)[0], axis=-1)
            y = tf.einsum('np,in->ip', t, y)
        else:
            y = tf.embedding_lookup(t, x)
        a = self.adapt_ws[i]
        if a is not None:
            y = tf.einsum('ip,ph->ih', y, a)
        return y


class TypEmbed(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items(
            'dim_hidden',
            'tok_types',
        ))

    def build(self, input_shape):
        cfg = self.cfg
        self.typ_w = self.add_weight('typ_w', (cfg.tok_types, cfg.dim_hidden))
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    @tf.function
    def call(self, inputs, mask=None):
        x, typ = inputs
        y = typ
        if mask is not None:
            y *= tf.cast(mask, typ.dtype)
        y = tf.one_hot(y, self.cfg.tok_types)
        return x + tf.einsum('bie,eh->bih', y, self.typ_w)


class PosEmbed(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'dim_hidden',
                'len_src',
                'len_tgt',
                'pos_max_len',
            ))

    def build(self, input_shape):
        cfg = self.cfg
        p = max(cfg.pos_max_len or 0, cfg.len_src, cfg.len_tgt)
        self.pos_b = self.add_weight('pos_b', (p, cfg.dim_hidden))
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask=None):
        x = inputs
        y = self.pos_b[:tf.shape[1], :]
        if mask is not None:
            y *= tf.cast(mask, self.pos_b.dtype)
        return x + y


class PosTiming(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'dim_hidden',
                'pos_max',
                'pos_min',
                'pos_start',
                'len_src',
                'len_tgt',
                'pos_max_len',
            ))

    def build(self, input_shape):
        cfg = self.cfg
        d = cfg.dim_hidden
        p = max(cfg.pos_max_len or 0, cfg.len_src, cfg.len_tgt)
        a = (cfg.pos_max, cfg.pos_min, cfg.pos_start)
        self.pos_b = utils.pos_timing_2(d, p, *a)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, mask=None):
        x = inputs
        y = self.pos_b
        if mask is not None:
            y *= tf.cast(mask, self.pos_b.dtype)
        return x + y


class RelEmbed(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'dim_hidden',
                'len_src',
                'len_tgt',
                'pos_max_len',
            ))

    def build(self, input_shape):
        cfg = self.cfg
        p = max(cfg.pos_max_len or 0, cfg.len_src, cfg.len_tgt)
        self.pos_b = utils.pos_timing(cfg.dim_hidden, p)

    @tf.function
    def call(self, inputs, mask=None):
        x = inputs
        y = self.pos_b
        if mask is not None:
            y *= tf.cast(mask, self.pos_b.dtype)
        return x + y
