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

import qnarre.neura.utils as U

from qnarre.neura.layers.search import Beam
from qnarre.neura.layers.norm import LayerNorm, PreProc, PostProc
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming

from qnarre.neura.layers.attn import Attn
from qnarre.neura.layers.ffnet import FFNet


class Trafo:
    typ_emb = pos_emb = src_b = mem_b = beam = None

    def __init__(self, ps):
        self.ps = ps
        self.tok_emb = TokEmbed(ps)
        if ps.tok_types:
            self.typ_emb = TypEmbed(ps)
        if ps.pos_embed:
            p = PosEmbed(ps) if ps.pos_embed == 'embed' else None
            p = PosTiming(ps) if ps.pos_embed == 'timing' else p
            self.pos_emb = p
        self.norm = LayerNorm(ps)
        self.dropout = tf.Dropout(ps.drop_hidden)
        self.pre = PreProc(ps)
        self.post = PostProc(ps)
        self.enc_stack = EncStack(self)
        self.dec_stack = DecStack(self)
        self.logits = tf.Dense(ps.num_toks, activation=None)
        if ps.beam_size:
            self.beam = Beam(ps, lambda *a, **kw: self.to_logp(*a, **kw))

    @tf.function
    def __call__(self, inputs):
        ps = self.ps
        src, typ, tgt = inputs
        ctx, bias = self.encode(src, typ)
        if tgt is not None:
            logp, logi, unk = self.to_logp(tgt, ctx, bias)
            return logi

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

    def embed(self, tok, typ=None):
        y = self.tok_emb(tok)
        if typ is not None and self.typ_emb:
            y = self.typ_emb([y, typ])
        if self.pos_emb:
            y = self.pos_emb(y)
        y = self.norm(y)
        y = self.dropout(y)
        return y

    def encode(self, src, typ):
        ctx, bias = None, None
        if src is not None:
            y = self.embed(src, typ)
            ctx, bias = self.enc_stack(y)
        return ctx, bias

    def decode(self, tgt, ctx, bias):
        y = self.embed(tgt)
        y = self.dec_stack([y, ctx, bias])
        return y

    def to_logp(self, tgt, ctx, bias, i=None):
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


class Stack:
    prox_b = None

    @staticmethod
    def proximity(max_len):
        y = tf.range(max_len, dtype=tf.floatx())
        y = tf.expand_dims(y, axis=0) - tf.expand_dims(y, axis=1)
        y = -tf.log1p(tf.abs(y))
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        return y

    def __init__(self, owner):
        self.ps = owner.ps
        self.pre = owner.pre
        self.post = owner.post

    def attn_bias(self, mask):
        y = tf.logical_not(mask)
        y = tf.cast(y, tf.floatx()) * self.ps.big_neg
        y = tf.expand_dims(tf.expand_dims(y, axis=1), axis=3)
        return y


class EncStack(Stack):
    def __init__(self, owner):
        super().__init__(owner)
        ps = self.ps
        n = ps.layers_enc or ps.layers_stack
        self.encs = [Encoder(f'enc_{i}', owner) for i in range(n)]
        if ps.bias_prox:
            self.prox_b = self.proximity(ps.len_src)

    def __call__(self, inputs, mask):
        x = inputs
        ab = rb = self.attn_bias(mask)
        if self.prox_b is not None:
            rb += self.prox_b
        y = self.pre.drop(x)
        for e in self.encs:
            y = e([y, rb])
        y = self.post([x, y])
        return y, ab


class DecStack(Stack):
    def __init__(self, owner):
        super().__init__(owner)
        ps = self.ps
        n = ps.layers_dec or ps.layers_stack
        self.decs = [Decoder(f'dec_{i}', owner) for i in range(n)]
        if ps.prox_bias:
            self.prox_bias = self.proximity(ps.len_tgt)

    def __call__(self, inputs, mask):
        x, ctx, ab = inputs
        rb = self.attn_bias(mask[0])
        ps = self.ps
        if ps.causal_refl:
            if ps.prepend_mode == 'prepend_inputs_full_attention':
                y = tf.cumsum(tf.cumsum(rb, axis=1), axis=1)
                y2 = tf.expand_dims(y, axis=1)
                y = tf.greater(y2, tf.expand_dims(y, axis=2))
                b = tf.expand_dims(tf.cast(y, tf.floatx()) * -1e9, axis=1)
            else:
                ln = tf.int_shape(x)[1]
                sh = (1, 1, ln, ln)
                b = U.ones_band_part(ln, ln, -1, 0, out_shape=sh)
                b = -1e9 * (1.0 - b)
        if self.prox_bias:
            rb += self.prox_bias
        y = self.pre.drop(x)
        for d in self.decs:
            y = d([y, rb, ctx, ab])
        y = self.post([x, y])
        return y


class Encoder:
    def __init__(self, name, owner):
        self.refl = Attn(owner, name=name + '_refl')
        self.ffnet = FFNet(owner, name=name + '_ffnet')

    def __call__(self, inputs):
        x, rb = inputs
        y = self.refl([x, x, rb])
        y = self.ffnet(y)
        return y


class Decoder:
    def __init__(self, name, owner):
        self.refl = Attn(owner, name=name + '_refl')
        self.attn = Attn(owner, name=name + '_attn')
        self.ffnet = FFNet(owner, name=name + '_ffnet')

    def __call__(self, inputs):
        x, rb, ctx, ab = inputs
        y = self.refl([x, x, rb])
        y = self.attn([y, ctx, ab])
        y = self.ffnet(y)
        return y
