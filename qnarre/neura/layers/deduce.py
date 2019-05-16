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
from qnarre.neura.layers import base


class Layer(base.Layer):
    def add_weight(self, *pa, **kw):
        return super().add_weight(*pa, dtype=tf.floatx(), **kw)

    def add_bias(self, *pa, **kw):
        return super().add_bias(*pa, dtype=tf.floatx(), **kw)


class TokDeduce(Layer):
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

    def __init__(self, ps, table_ws, adapt_ws=None, **kw):
        super().__init__(ps, **kw)
        self._compute_output_and_mask_jointly = True
        self.adapt_ws = adapt_ws or []
        self.table_ws = table_ws
        self.table_bs = []

    def build(self, input_shape):
        cfg = self.cfg
        h = cfg.dim_hidden
        d = cfg.dim_embed or h
        n = len(cfg.brackets)
        if n:
            self.clust_w = self.add_weight(f'clust_w', (n, d))
            self.clust_b = self.add_bias(f'clust_b', (n, ))
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            t = self.add_bias(f'table_b{i}', (e - b, ))
            self.table_bs.append(t)
            if len(self.adapt_ws) == i:
                a = None
                p = d // (len(bs)**i)
                if p != h:
                    a = self.add_weight(f'adapt_w{i}', (p, h))
                self.adapt_ws.append(a)
            b = e
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        x, _ = inputs
        return tf.not_equal(x, 0)

    def compute_output_shape(self, input_shape):
        x, _ = input_shape
        return x

    @tf.function
    def call(self, inputs):
        cfg = self.cfg
        x, ctx = inputs
        mask = tf.not_equal(x, cfg.PAD)
        if cfg.brackets:
            y = tf.zeros_like(x, dtype=tf.floatx())
            bs = (cfg.brackets or []) + [cfg.num_toks]
            b = 0
            for i, e in enumerate(bs):
                m = (x >= (b or 1)) & (x < e)
                tgt = tf.boolean_mask(x, m) - b

                def gather(logp):
                    r = tf.range(tf.shape(logp)[0])
                    return tf.gather_nd(logp, tf.stack([r, tgt], 1))

                if i == 0:
                    p = self.logits(ctx, i)
                    p = tf.log_softmax(p)
                    mp = tf.boolean_mask(p, m)
                    u = gather(mp)
                else:
                    c = tf.boolean_mask(ctx, m)
                    c = tf.squeeze(self.logits(c[None]), 0)
                    c = tf.log_softmax(c)
                    mp = tf.boolean_mask(p, m)
                    u = mp[:, bs[1] + -1] + gather(c)
                y = tf.tensor_scatter_nd_add(y, tf.where(m), -u)
        else:
            y = self.logits(ctx)
            y = tf.sparse_softmax_cross_entropy_with_logits(labels=x, logits=y)
        y = tf.reduce_mean(y)
        y._keras_mask = mask
        return y

    def logits(self, x, i=None):
        y = x
        a = self.adapt_ws[i or 0]
        if a is not None:
            y = tf.einsum('ih,eh->ie', y, a)
        t = self.table_ws[i or 0]
        b = self.table_bs[i or 0]
        if i == 0:
            t = tf.concat([t, self.clust_w], 0)
            b = tf.concat([b, self.clust_b], 0)
        y = tf.einsum('ie,ne->in', y, t) + b
        return y
