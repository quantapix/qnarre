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
from qnarre.neura.layers.search import Beam


class Deduce(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'END',
                'PAD',
                'UNK',
                'batch_size',
                'beam_size',
                'brackets',
                'dim_embed',
                'dim_hidden',
                'emb_one_hot',
                'num_toks',
                'share_adapt',
                'share_table',
            ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        self.table_ws = owner.tok_emb.table_ws if cfg.share_table else []
        self.table_bs = []
        self.adapt_ws = owner.tok_emb.adapt_ws if cfg.share_adapt else []

    def build(self, input_shape):
        cfg = self.cfg
        h = cfg.dim_hidden
        d = cfg.dim_embed or h
        bs = cfg.brackets or []
        n = len(bs)
        if n:
            self.clust_w = self.add_weight(f'clust_w', (n, d))
            self.clust_b = self.add_bias(f'clust_b', (n, ))
        bs += [cfg.num_toks]
        b = 0
        assert b == cfg.PAD
        for i, e in enumerate(bs):
            p = d // (len(bs)**i)
            if len(self.table_ws) == i:
                t = self.add_weight(f'table_w{i}', (e - b, p))
                self.table_ws.append(t)
            t = self.add_bias(f'table_b{i}', (e - b, ))
            self.table_bs.append(t)
            if len(self.adapt_ws) == i:
                a = None if p == h else self.add_weight(f'adapt_w{i}', (p, h))
                self.adapt_ws.append(a)
            b = e
        self.one_hot = cfg.emb_one_hot
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @tf.function
    def call(self, inputs):
        cfg = self.cfg
        x, tgt = inputs
        if cfg.brackets:
            y = tf.zeros_like(tgt, dtype=tf.floatx())
            bs = cfg.brackets + [cfg.num_toks]
            b = 0
            for i, e in enumerate(bs):
                msk = (tgt >= (b or 1)) & (tgt < e)
                mt = tf.boolean_mask(tgt, msk) - b
                gi = tf.stack([tf.range(tf.shape(mt)[0]), mt])
                if i == 0:
                    logp = tf.log_softmax(self.logits(x, i))
                    mp = tf.boolean_mask(logp, msk)
                    u = tf.gather_nd(mp, gi)
                else:
                    mp = tf.boolean_mask(logp, msk)
                    u = mp[:, bs[i - 1]]
                    mc = tf.boolean_mask(x, msk)[None]
                    mp = tf.log_softmax(self.logits(mc, i))
                    mp = tf.squeeze(mp, 0)
                    u += tf.gather_nd(mp, gi)
                y = tf.tensor_scatter_nd_add(y, tf.where(msk), -u)
                b = e
        else:
            y = self.logits(x)
            # f = tf.SparseCategoricalAccuracy
            # self.add_metric(f(name='acc')(tgt, y))
            f = tf.sparse_softmax_cross_entropy_with_logits
            loss = f(labels=tgt, logits=y)
        # self.add_loss(lambda: tf.reduce_mean(loss))
        return y

    def logits(self, x, i=None):
        y = x
        a = self.adapt_ws[i or 0]
        if a is not None:
            y = tf.einsum('bih,ph->bip', y, a)
        t = self.table_ws[i or 0]
        b = self.table_bs[i or 0]
        if i == 0:
            t = tf.concat([t, self.clust_w], 0)
            b = tf.concat([b, self.clust_b], 0)
        y = tf.einsum('bie,ne->bin', y, t) + b
        return y


class Search(Deduce):
    beam = None

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, owner, **kw)
        cfg = self.cfg
        if cfg.beam_size:
            self.beam = Beam(ps, self, name='beam')

    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
    def call(self, inputs):
        x, ctx = inputs
        cfg = self.cfg
        return x
    """
        if self.beam is not None:
            tgt, score = self.beam([x, ctx])
        else:
            logp, logi, unk = self.search(tgt, ctx)
            sh = tgt.shape
            b = tf.range(cfg.batch_size)
            for i in range(sh[-1]):
                if tf.reduce_any(unk[:, i]) is True:
                    y = tf.argmax(logp[:, i, :], axis=1, output_type=tf.int32)
                    ii = tf.constant([i] * cfg.batch_size)
                    sel = tf.stack([b, ii])
                    tgt = tf.tensor_scatter_nd_update(tgt, sel, y)
                    e = tf.equal(tgt, cfg.END)
                    if tf.reduce_all(tf.reduce_any(e, axis=1)) is True:
                        break
                    logp, logi, unk = self.to_logp(tgt, ctx, i)
        return tf.one_hot(tgt, cfg.num_toks, 0.0, utils.big_neg)
    """

    def search(self, tgt, ctx, i=None):
        cfg = self.cfg
        unk = tf.equal(tgt, cfg.UNK)
        prior = tf.one_hot(tgt, cfg.num_toks, 0.0, utils.big_neg)
        if i is not None:
            unk = unk[:, i]
            prior = prior[:, i, :]
        if tf.reduce_all(unk) is True:
            logi = prior
        else:
            y = self.decode(tgt, ctx)
            if i is not None:
                y = y[:, i, :]
            sh = y.shape  # tf.int_shape(y)
            y = tf.reshape(y, (-1, sh[-1]))
            y = self.logits(y)
            y = tf.reshape(y, sh[:-1] + y.shape[-1:])
            u = tf.expand_dims(unk, axis=2)
            u = tf.broadcast_to(u, y.shape)
            logi = tf.where(u, y, prior)
        logp = y - tf.reduce_logsumexp(y, axis=-1, keepdims=True)
        return logp, logi, unk
