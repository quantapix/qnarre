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

from qnarre.neura.layers.base import Layer


class DeduceLoss(Layer):
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
                'share_adapt',
                'share_table',
            ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        self.table_ws = owner.embed.table_ws if cfg.share_table else []
        self.table_bs = []
        self.adapt_ws = owner.embed.adapt_ws if cfg.share_adapt else []

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
        return tf.TensorShape(())

    @tf.function
    def call(self, inputs):
        cfg = self.cfg
        x, ctx = inputs
        if cfg.brackets:
            y = tf.zeros(tf.int_shape(x) + (cfg.dim_hidden, ))
            bs = cfg.brackets + [cfg.num_toks]
            b = 0
            for i, e in enumerate(bs):
                msk = (x >= (b or 1)) & (x < e)
                tgt = tf.boolean_mask(x, msk) - b
                gi = tf.stack([tf.range(tf.shape(tgt)[0]), tgt])
                if i == 0:
                    logp = tf.log_softmax(self.logits(ctx, i))
                    mp = tf.boolean_mask(logp, msk)
                    u = tf.gather_nd(mp, gi)
                else:
                    mp = tf.boolean_mask(logp, msk)
                    u = mp[:, bs[i - 1]]
                    mc = tf.boolean_mask(ctx, msk)[None]
                    mp = tf.log_softmax(self.logits(mc, i))
                    mp = tf.squeeze(mp, 0)
                    u += tf.gather_nd(mp, gi)
                y = tf.tensor_scatter_nd_add(y, tf.where(msk), -u)
                b = e
        else:
            y = self.logits(ctx)
            y = tf.sparse_softmax_cross_entropy_with_logits(labels=x, logits=y)
        y = tf.reduce_mean(y)
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


class DeduceToks(DeduceLoss):
    @tf.function
    def call(self, inputs):
        cfg = self.cfg

    def deduce(self, tgt, ctx, bias, i=None):
        ps = self.ps
        unk = tf.equal(tgt, ps.UNK)
        prior = tf.one_hot(tgt, ps.num_toks, 0.0, ps.big_neg)
        if i is not None:
            unk = unk[:, i]
            prior = prior[:, i, :]
        if tf.reduce_all(unk) is True:
            logi = prior
        else:
            y = self.decode(tgt, ctx, bias)
            if i is not None:
                y = y[:, i, :]
            sh = tf.int_shape(y)
            y = tf.reshape(y, (-1, sh[-1]))
            y = self.logits(y)
            y = tf.reshape(y, sh[:-1] + tf.int_shape(y)[-1:])
            u = tf.expand_dims(unk, axis=2)
            u = tf.broadcast_to(u, tf.int_shape(y))
            logi = tf.where(u, y, prior)
        logp = y - tf.reduce_logsumexp(y, axis=-1, keepdims=True)
        return logp, logi, unk

    """
    def call(self, inputs, training=None, **kw):
        ps = self.ps
        src, typ, tgt = inputs
        ctx, bias = self.encode(src, typ, **kw)
        if tgt is not None:
            if training is not None and self.beam is not None:
                tgt, score = self.beam([tgt, ctx, bias], **kw)
            else:
                logp, logi, unk = self.to_logp(tgt, ctx, bias, **kw)
                sh = tf.int_shape(tgt)
                b = tf.range(ps.batch_size)
                for i in range(sh[-1]):
                    if tf.reduce_any(unk[:, i]) is True:
                        y = tf.argmax(logp[:, i, :],
                                     axis=1,
                                     output_type=tf.int32)
                        ii = tf.constant([i] * ps.batch_size)
                        sel = tf.stack([b, ii])
                        tgt = tf.tensor_scatter_nd_update(tgt, sel, y)
                        e = tf.equal(tgt, ps.END)
                        if tf.reduce_all(tf.reduce_any(e, axis=1)) is True:
                            break
                        logp, logi, unk = self.to_logp(tgt, ctx, bias, **kw)
            return tf.one_hot(tgt, ps.num_toks, 0.0, ps.big_neg)
    """
