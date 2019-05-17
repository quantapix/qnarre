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
from qnarre.neura.layers.attn import Attn
from qnarre.neura.layers.base import Layer
from qnarre.neura.layers.ffnet import FFNet
from qnarre.neura.layers.search import Beam
from qnarre.neura.layers.norm import Norm, PreProc, PostProc
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming


class Trafo(Layer):
    typ_emb = pos_emb = src_b = mem_b = beam = None

    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'beam_size',
                'drop_hidden',
                'num_toks',
                'pos_emb',
                'tok_typs',
            ))

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        self.tok_emb = TokEmbed(ps, name='tok_emb')
        if cfg.tok_typs:
            self.typ_emb = TypEmbed(ps, name='typ_emb')
        if cfg.pos_emb == 'embed':
            self.pos_emb = PosEmbed(ps, name='pos_emb')
        elif cfg.pos_emb == 'timing':
            self.pos_emb = PosTiming(ps, name='time_emb')
        else:
            assert cfg.pos_emb is None
        self.pre = PreProc(ps, name='pre_proc')
        self.post = PostProc(ps, name='post_proc')
        self.enc_stack = EncStack(ps, self, name='enc_stack')
        self.dec_stack = DecStack(ps, self, name='dec_stack')
        self.out = tf.Dense(cfg.num_toks, name='out', activation=None)
        if cfg.beam_size:
            self.beam = Beam(ps, self, name='beam')

    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
    def call(self, inputs):
        src, typ, s_mems, tgt, t_mems, ctx = inputs
        out = s_ms = t_ms = None
        if src is not None:
            ctx, s_ms = self.encode(src, typ, s_mems)
        if tgt is not None:
            out, t_ms = self.deduce(tgt, t_mems, ctx)
        return [out, s_ms, t_ms]

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

    def embed(self, x, typ=None):
        y = self.tok_emb(x)
        if typ is not None and self.typ_emb:
            y = self.typ_emb([y, typ])
        if self.pos_emb:
            y = self.pos_emb(y)
        return y

    def encode(self, src, typ, mems):
        y = self.embed(src, typ)
        y, ms = self.enc_stack([y, mems])
        return y, ms

    def decode(self, tgt, mems, ctx):
        y = self.embed(tgt)
        y, ms = self.dec_stack([y, mems, ctx])
        return y, ms

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


class Stack(Layer):
    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        self.pre = owner.pre
        self.post = owner.post

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        x, ms = input_shape[:2]
        ms = ([x] * len(self.encs)) if ms is None else ms
        return [x, ms]

    def new_mem(self, x, old):
        mlen = self.cfg.len_mem
        if mlen is None or old is None:
            m = x
        elif mlen == 0:
            return old
        else:
            m = tf.concat([old, x], 0)[-mlen:]
        return tf.stop_gradient(m)


class EncStack(Stack):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items(
            'len_mem',
            'num_enc_lays',
            'num_stack_lays',
        ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, owner, **kw)
        cfg = self.cfg
        n = cfg.num_enc_lays or cfg.num_stack_lays
        self.encs = [Encoder(ps, owner, f'enc_{i}') for i in range(n)]

    @tf.function
    def call(self, inputs):
        x, mems = inputs
        y = self.pre(x)
        ms = []
        for i, e in enumerate(self.encs):
            m = None if mems is None else mems[i]
            ms.append(self.new_mem(y, m))
            y = e([y, m])
        y = self.post([x, y])
        return y, ms


class DecStack(Stack):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items(
            'len_mem',
            'num_dec_lays',
            'num_stack_lays',
        ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, owner, **kw)
        cfg = self.cfg
        n = cfg.num_dec_lays or cfg.num_stack_lays
        self.decs = [Decoder(ps, owner, f'dec_{i}') for i in range(n)]

    @tf.function
    def call(self, inputs):
        x, mems, ctx = inputs
        """
        cfg = self.cfg
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
        """
        y = self.pre(x)
        ms = []
        for i, d in enumerate(self.decs):
            m = None if mems is None else mems[i]
            ms.append(self.new_mem(y, m))
            y = d([y, m, ctx])
        y = self.post([x, y])
        return y, ms


class EncDec(Layer):
    def __init__(self, ps, owner, name, **kw):
        super().__init__(ps, **kw)
        self.refl = Attn(ps, owner, name=name + '_refl')
        self.ffnet = FFNet(ps, owner, name=name + '_ffnet')

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Encoder(EncDec):

    @tf.function
    def call(self, inputs):
        x, mem = inputs
        y = self.refl([x, mem])
        y = self.ffnet(y)
        return y


class Decoder(Layer):
    def __init__(self, ps, owner, name, **kw):
        super().__init__(ps, owner, name, **kw)
        self.attn = Attn(ps, owner, name=name + '_attn')

    @tf.function
    def call(self, inputs):
        x, mem, ctx = inputs
        y = self.refl([x, mem])
        y = self.attn([y, ctx])
        y = self.ffnet(y)
        return y
