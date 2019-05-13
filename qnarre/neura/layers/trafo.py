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

from qnarre.neura.layers.ffn import ffns
from qnarre.neura.layers.attn import attns


class Trafo(tf.Layer):
    typ_embed, pos_embed, beam = None, None, None

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.tok_embed = TokEmbed(PS)
        if PS.token_types:
            self.typ_embed = TypEmbed(PS)
        if PS.pos_embed:
            p = PosEmbed(PS) if PS.pos_embed == 'embed' else None
            p = PosTiming(PS) if PS.pos_embed == 'timing' else p
            self.pos_embed = p
        self.norm = LayerNorm(PS)
        self.drop = tf.Dropout(PS.hidden_drop)
        self.pre = PreProc(PS)
        self.post = PostProc(PS)
        self.enc_stack = EncStack(PS, self.pre, self.post)
        self.dec_stack = DecodeStack(PS, self.pre, self.post)
        self.logits = tf.Dense(PS.vocab_size, activation=None)
        if PS.beam_size:
            self.beam = Beam(PS, lambda *a, **kw: self.to_logp(*a, **kw))

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, training=None, **kw):
        PS = self.PS
        src, typ, tgt = inputs
        ctx, bias = self.encode(src, typ, **kw)
        if tgt is not None:
            logp, logi, unk = self.to_logp(tgt, ctx, bias, **kw)
            return logi

    """
    def call(self, inputs, training=None, **kw):
        PS = self.PS
        src, typ, tgt = inputs
        ctx, bias = self.encode(src, typ, **kw)
        if tgt is not None:
            if training is not None and self.beam is not None:
                tgt, score = self.beam([tgt, ctx, bias], **kw)
            else:
                logp, logi, unk = self.to_logp(tgt, ctx, bias, **kw)
                sh = tf.int_shape(tgt)
                b = tf.range(PS.batch_size)
                for i in range(sh[-1]):
                    if tf.reduce_any(unk[:, i]) is True:
                        y = tf.argmax(logp[:, i, :],
                                     axis=1,
                                     output_type=tf.int32)
                        ii = tf.constant([i] * PS.batch_size)
                        sel = tf.stack([b, ii])
                        tgt = tf.tensor_scatter_nd_update(tgt, sel, y)
                        e = tf.equal(tgt, PS.END)
                        if tf.reduce_all(tf.reduce_any(e, axis=1)) is True:
                            break
                        logp, logi, unk = self.to_logp(tgt, ctx, bias, **kw)
            return tf.one_hot(tgt, PS.vocab_size, 0.0, PS.big_neg)
    """

    def get_config(self):
        c = super().get_config()
        c['PS'] = self.PS
        return c

    def embed(self, tok, typ=None, **kw):
        y = self.tok_embed(tok, **kw)
        if typ is not None and self.typ_embed:
            y = self.typ_embed([y, typ], **kw)
        if self.pos_embed:
            y = self.pos_embed(y, **kw)
        y = self.norm(y, **kw)
        y = self.drop(y, **kw)
        return y

    def encode(self, src, typ, **kw):
        ctx, bias = None, None
        if src is not None:
            y = self.embed(src, typ, **kw)
            ctx, bias = self.enc_stack(y, **kw)
        return ctx, bias

    def decode(self, tgt, ctx, bias, **kw):
        y = self.embed(tgt, **kw)
        y = self.dec_stack([y, ctx, bias], **kw)
        return y

    def to_logp(self, tgt, ctx, bias, i=None, **kw):
        PS = self.PS
        unk = tf.equal(tgt, PS.UNK)
        prior = tf.one_hot(tgt, PS.vocab_size, 0.0, PS.big_neg)
        if i is not None:
            unk = unk[:, i]
            prior = prior[:, i, :]
        if tf.reduce_all(unk) is True:
            logi = prior
        else:
            y = self.decode(tgt, ctx, bias, **kw)
            if i is not None:
                y = y[:, i, :]
            sh = tf.int_shape(y)
            y = tf.reshape(y, (-1, sh[-1]))
            y = self.logits(y, **kw)
            y = tf.reshape(y, sh[:-1] + tf.int_shape(y)[-1:])
            u = tf.expand_dims(unk, axis=2)
            u = tf.broadcast_to(u, tf.int_shape(y))
            logi = tf.where(u, y, prior)
        logp = y - tf.reduce_logsumexp(y, axis=-1, keepdims=True)
        return logp, logi, unk


class Stack(tf.Layer):
    prox_bias = None

    @staticmethod
    def proximity(max_len):
        y = tf.range(max_len, dtype=tf.floatx())
        y = tf.expand_dims(y, axis=0) - tf.expand_dims(y, axis=1)
        y = -tf.log1p(tf.abs(y))
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        return y

    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post

    def attn_bias(self, mask):
        y = tf.logical_not(mask)
        y = tf.cast(y, tf.floatx()) * self.PS.big_neg
        y = tf.expand_dims(tf.expand_dims(y, axis=1), axis=3)
        return y


class EncStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.enc_layers or PS.stack_layers
        self.encs = [Encoder(*a, name=f'enc_{i}') for i in range(n)]

    def build(self, input_shape):
        # if self.PS.prox_bias:
        #     self.prox_bias = self.proximity(input_shape[1])
        return super().build(input_shape)

    def call(self, inputs, mask, **kw):
        x = inputs
        ab = rb = self.attn_bias(mask)
        if self.prox_bias:
            rb += self.prox_bias
        y = self.pre.drop(x, **kw)
        for e in self.encs:
            y = e([y, rb], **kw)
        y = self.post([x, y], **kw)
        return y, ab


class DecodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.dec_layers or PS.stack_layers
        self.decs = [Decoder(*a, name=f'dec_{i}') for i in range(n)]

    def build(self, input_shape):
        # if self.PS.prox_bias:
        #     self.prox_bias = self.proximity(input_shape[0][1])
        return super().build(input_shape)

    def call(self, inputs, mask, **kw):
        x, ctx, ab = inputs
        rb = self.attn_bias(mask[0])
        PS = self.PS
        if PS.causal_refl:
            if PS.prepend_mode == 'prepend_inputs_full_attention':
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
        y = self.pre.drop(x, **kw)
        for d in self.decs:
            y = d([y, rb, ctx, ab], **kw)
        y = self.post([x, y], **kw)
        return y


class Encoder(tf.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = attns[PS.refl_type](*a)
        self.ffn = ffns[PS.ffn_type](*a)

    def call(self, inputs, **kw):
        x, rb = inputs
        y = self.refl([x, x, rb], **kw)
        y = self.ffn(y, **kw)
        return y


class Decoder(tf.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = attns[PS.refl_type](*a)
        self.attn = attns[PS.attn_type](*a)
        self.ffn = ffns[PS.ffn_type](*a, conv_pad='LEFT')

    def call(self, inputs, **kw):
        x, rb, ctx, ab = inputs
        y = self.refl([x, x, rb], **kw)
        y = self.attn([y, ctx, ab], **kw)
        y = self.ffn(y, **kw)
        return y
