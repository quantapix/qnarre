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

import numpy as N
import tensorflow as T

import qnarre.neura.utils as U

from qnarre.neura.layers.norm import LayerNorm
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming

KS = T.keras
K = KS.backend
KL = KS.layers


class Trafo(KL.Layer):
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
        self.norm = LayerNorm()
        self.drop = KL.Dropout(PS.hidden_drop)
        pre, post = PreProc(PS), PostProc(PS)
        self.enc_stack = EncodeStack(PS, pre, post)
        self.dec_stack = DecodeStack(PS, pre, post)
        self.logits = KL.Dense(PS.vocab_size, activation=None)

    def build(self, input_shape):
        _, tgt = input_shape
        kw = dict(dtype='int32', trainable=False)
        self.tok_out = self.add_weight(shape=tgt[:2], **kw)
        kw.update(dtype='bool')
        kw.update(initializer='zeros')
        self.mlm_bias = self.add_weight(shape=self.PS.vocab_size, **kw)
        self.bias = self.add_weight(shape=2, **kw)
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        e, d = None, None
        src, tgt = input_shape
        if src:
            e, _ = self.enc_stack.output_shape
        if tgt:
            d = self.logits.output_shape
        return e, d

    def embed(self, ins, **kw):
        tok, typ = ins
        y = self.tok_embed(tok, **kw)
        if self.typ_embed:
            y = self.typ_embed([y, typ], **kw)
        if self.pos_embed:
            y = self.pos_embed(y, **kw)
        y = self.norm(y, **kw)
        return self.drop(y, **kw)

    def encode(self, ins, **kw):
        y, b = None, None
        if ins:
            y = self.embed(ins, **kw)
            y, b = self.enc_stack(y, **kw)
        return y, b

    def decode(self, ins, ctx, att, **kw):
        y = None
        if ins:
            y = self.embed(ins, **kw)
            y = self.dec_stack([ctx, att, y], **kw)
        return y

    def to_logits(self, x, unks=None, prior=None, **kw):
        xs = K.int_shape(x)
        y = K.reshape(x, (-1, xs[-1]))
        y = self.logits(y, **kw)
        ys = K.int_shape(y)
        y = K.reshape(y, xs[:-1] + ys[-1:])
        if unks:
            y = T.where(unks, y, prior)
        return y

    def to_toks(self, x, **kw):
        pass

    def sample(self, x):
        t = self.PS.sampling_temp or 0.0
        if self.PS.sampling_method == 'argmax':
            t = 0.0
        keep_top_k = self.PS.keep_top_k or -1
        if t == 0.0:
            # TF argmax doesn't handle >5 dimensions, so we reshape here.
            sh = K.int_shape(x)
            argmax = T.argmax(T.reshape(x, [-1, sh[-1]]), axis=1)
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

    def call(self, inputs, training=None, **kw):
        src, tgt = inputs
        ctx, att = self.encode(src, **kw)
        if training:
            y = self.decode(tgt, ctx, att, **kw)
            if y:
                y = self.to_logits(y, **kw)
                return self.to_toks(y, **kw)
            return ctx, att
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
                    d = i >= K.int_shape(tgt)[-1]
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


class Stack(KL.Layer):
    prox_bias = None

    @staticmethod
    def proximity(slen):
        p = K.arange(slen, dtype=K.floatx())
        p = K.expand_dims(p, 0) - K.expand_dims(p, 1)
        return K.expand_dims(K.expand_dims(-T.math.log1p(K.abs(p)), 0), 0)

    @staticmethod
    def attn_bias(mask):
        f = K.floatx()
        fmin = T.float16.min if f == 'float16' else T.float32.min
        b = K.cast(mask, f) * fmin
        return K.expand_dims(K.expand_dims(b, axis=1), axis=1)

    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post


class EncodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.encode_layers or PS.stack_layers
        self.encs = [Encoder(*a, name=f'enc_{i}') for i in range(n)]

    def build(self, input_shape):
        if self.PS.prox_bias:
            self.prox_bias = self.proximity(input_shape[1])
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.encs[-1].output_shape

    def call(self, inputs, mask, **kw):
        x = inputs
        b = self.attn_bias(mask)
        if self.prox_bias:
            b += self.prox_bias
        # if self.PS.pad_remover:
        #     kw.update(pad_remover=U.PadRemover(mask))
        y = self.pre.drop(x, **kw)
        for e in self.encs:
            y = e([y, b], **kw)
        return self.post([x, y], **kw), b


class DecodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.decode_layers or PS.stack_layers
        self.decs = [Decoder(*a, name=f'dec_{i}') for i in range(n)]

    def build(self, input_shape):
        if self.PS.prox_bias:
            self.prox_bias = self.proximity(input_shape[2][1])
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.decs[-1].output_shape

    def call(self, inputs, mask, **kw):
        x, b, t = inputs
        sb = self.attn_bias(mask[2])
        PS = self.PS
        if PS.causal_refl:
            if PS.prepend_mode == 'prepend_inputs_full_attention':
                p = K.cumsum(K.cumsum(sb, axis=1), axis=1)
                p = K.greater(K.expand_dims(p, 1), K.expand_dims(p, 2))
                sb = K.expand_dims(K.cast(p, K.floatx()) * -1e9, 1)
            else:
                ln = K.int_shape(t)[1]
                sh = (1, 1, ln, ln)
                b = U.ones_band_part(ln, ln, -1, 0, out_shape=sh)
                sb = -1e9 * (1.0 - b)
        if self.prox_bias:
            sb += self.prox_bias
        # y = T.pad(t, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # t = T.concat([pad_value, t], axis=1)[:, :-1, :]
        y = self.pre.drop(t, **kw)
        for d in self.decs:
            y = d([x, b, y, sb], **kw)
        return K.expand_dims(self.post([t, y], **kw), axis=2)


class Encoder(KL.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = _attns[PS.refl_type](*a)
        self.ffn = _ffns[PS.ffn_type](*a)

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        x, b = inputs
        y = self.refl([x, x, b], **kw)
        return self.ffn(y, **kw)


class Decoder(KL.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = _attns[PS.refl_type](*a)
        self.attn = _attns[PS.attn_type](*a)
        self.ffn = _ffns[PS.ffn_type](*a, conv_pad='LEFT')

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        x, b, t, sb = inputs
        y = self.refl([t, t, sb], **kw)
        y = self.attn([y, x, b], **kw)
        return self.ffn(y, **kw)
