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

import qnarre.neura as Q
import qnarre.neura.utils as U

from qnarre.neura.layers.ffn import ffns
from qnarre.neura.layers.attent import attns
from qnarre.neura.layers.norm import LayerNorm, PreProc, PostProc
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming


class Trafo(Q.Layer):
    typ_embed, pos_embed = None, None

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
        self.drop = Q.Dropout(PS.hidden_drop)
        self.pre = PreProc(PS)
        self.post = PostProc(PS)
        self.enc_stack = EncStack(PS, self.pre, self.post)
        self.dec_stack = DecodeStack(PS, self.pre, self.post)
        self.logits = Q.Dense(PS.vocab_size, activation=None)

    def build(self, input_shape):
        src, _, tgt = input_shape
        kw = dict(dtype='int32', trainable=False)
        # self.tok_out = self.add_weight(shape=tgt[:2], **kw)
        kw.update(dtype='bool', initializer='zeros')
        # self.mlm_bias = self.add_weight(shape=self.PS.vocab_size, **kw)
        # self.bias = self.add_weight(shape=2, **kw)
        return super().build(input_shape)

    """
    def compute_output_shape(self, input_shape):
        s = None
        src, _, tgt = input_shape
        if src:
            s = self.enc_stack.output_shape
        if tgt:
            s = self.logits.output_shape
        return s
    """

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
        y = None
        if tgt is not None:
            y = self.embed(tgt, **kw)
            y = self.dec_stack([y, ctx, bias], **kw)
        return y

    def to_logits(self, x, unks=None, prior=None, **kw):
        xs = Q.int_shape(x)
        y = Q.reshape(x, (-1, xs[-1]))
        y = self.logits(y, **kw)
        ys = Q.int_shape(y)
        y = Q.reshape(y, (-1, ) + xs[1:-1] + ys[-1:])
        if unks:
            y = Q.where(unks, y, prior)
        return y

    def to_toks(self, x, **kw):
        pass

    def sample(self, x):
        t = self.PS.sampling_temp or 0.0
        if self.PS.sampling_method == 'argmax':
            t = 0.0
        keep_top_k = self.PS.keep_top_k or -1
        """
        if t == 0.0:
            # TF argmax doesn't handle >5 dimensions, so we reshape here.
            sh = Q.int_shape(x)
            argmax = Q.argmax(T.reshape(x, [-1, sh[-1]]), axis=1)
            return T.reshape(argmax, sh[:-1])
        assert t > 0.0
        if keep_top_k != -1:
            if keep_top_k <= 0:
                raise ValueError("keep_top_k must either be -1 or positive.")

            vocab_size = shape_list(logits)[1]

            k_largest = T.contrib.nn.nth_element(logits,
                                                 n=keep_top_k,
                                                 reverse=True)
            k_largest = T.tile(T.reshape(k_largest, [-1, 1]), [1, vocab_size])

            # Force every position that is not in the top k to have probability near
            # 0 by setting the logit to be very negative.
            logits = T.where(T.less_equal(logits, k_largest),
                             T.ones_like(logits) * -1e6, logits)

        reshaped_logits = (T.reshape(logits, [-1, shape_list(logits)[-1]]) / t)
        choices = T.multinomial(reshaped_logits, 1)
        choices = T.reshape(choices,
                            shape_list(logits)[:logits.get_shape().ndims - 1])
        return choices
        """

    def call(self, inputs, training=None, **kw):
        src, typ, tgt = inputs
        ctx, bias = self.encode(src, typ, **kw)
        y = self.decode(tgt, ctx, bias, **kw)
        # if d:
        y = self.to_logits(y, **kw)
        # if training:
        return y
        # return self.to_toks(y, **kw)
        # if training:
        #    return c
        """
        if tgt:
            PS = self.PS
            if PS.beam_size > 1:
                toks = T.identity(tgt)
                return
            else:
                toks = tgt[0]
                unks = T.equal(toks, PS.UNK)
                prior = T.one_hot(toks, PS.vocab_size, 0.0, -1e9)
                eos = T.fill(toks[:1], False)
                scores = T.zeros(toks[:1])

                def not_done(i):
                    d = i >= Q.int_shape(tgt)[-1]
                    if eos:
                        d |= T.reduce_all(eos)
                    return T.logical_not(d)

                def loop(i):
                    y = self.decode(tgt, ctx, att, **kw)
                    y = self.to_logits(y, unks, prior)
                    lprb = y - T.reduce_logsumexp(y, axis=-1, keepdims=True)
                    t = self.sample(y[:, i, :])
                    nonlocal toks, unks, eos
                    toks = T.tensor_scatter_nd_update(toks, indices, t)
                    unks = T.tensor_scatter_nd_update(unks, indices, t)
                    eos |= T.equal(t, PS.EOS)
                    idx = T.stack([T.range(T.to_int64(PS.batch_size)), t],
                                  axis=1)
                    lprb += T.gather_nd(lprb, idx)
                    return i + 1

                _, toks, scores = T.while_loop(
                    not_done,
                    loop, [T.constant(0)],
                    shape_invariants=[T.TensorShape([])])
            return {"outputs": toks, "scores": scores}
        """


"""
            initial_ids = sos_id * T.ones([PS.batch_size], dtype=T.int32)
            decoded_ids, scores, cache = beam_search.beam_search(
                symbols_to_logits_fn,
                initial_ids,
                PS.beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=cache,
                eos_id=eos_id,
                stop_early=(PS.top_beams == 1))
            if PS.top_beams == 1:
                decoded_ids = decoded_ids[:, 0, 1:]
                scores = scores[:, 0]
            else:
                decoded_ids = decoded_ids[:, :PS.top_beams, 1:]
                scores = scores[:, :PS.top_beams]


    def get_config(self):
        c = super().get_config()
        c['PS'] = self.PS
        return c
"""


class Stack(Q.Layer):
    prox_bias = None

    @staticmethod
    def proximity(max_len):
        y = Q.arange(max_len, dtype=Q.floatx())
        y = Q.expand_dims(y, 0) - Q.expand_dims(y, 1)
        y = -Q.log1p(Q.abs(y))
        y = Q.expand_dims(Q.expand_dims(y, 0), 0)
        return y

    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post

    def attn_bias(self, mask):
        y = Q.logical_not(mask)
        y = Q.cast(y, Q.floatx()) * self.PS.big_neg
        y = Q.expand_dims(Q.expand_dims(y, axis=1), axis=3)
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
                p = Q.cumsum(Q.cumsum(rb, axis=1), axis=1)
                p = Q.greater(Q.expand_dims(p, 1), Q.expand_dims(p, 2))
                b = Q.expand_dims(Q.cast(p, Q.floatx()) * -1e9, 1)
            else:
                ln = Q.int_shape(x)[1]
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


class Encoder(Q.Layer):
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


class Decoder(Q.Layer):
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
