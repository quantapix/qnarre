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
from qnarre.neura.layers.norm import PreProc, PostProc
from qnarre.neura.layers.deduce import DeduceLoss, DeduceToks
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
                'pos_type',
                'tok_typs',
            ))

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        self.embed = TokEmbed(ps, name='embed')
        if cfg.tok_typs:
            self.typ_emb = TypEmbed(ps, name='typ_emb')
        if cfg.pos_type == 'embed':
            self.pos_emb = PosEmbed(ps, name='pos_emb')
        elif cfg.pos_type == 'timing':
            self.pos_emb = PosTiming(ps, name='time_emb')
        else:
            assert cfg.pos_type == 'relative'
        self.pre = PreProc(ps, name='pre_proc')
        self.post = PostProc(ps, name='post_proc')
        self.enc_stack = EncStack(ps, self, name='enc_stack')
        self.dec_stack = DecStack(ps, self, name='dec_stack')
        self.dedu_loss = DeduceLoss(ps, self, name='dedu_loss')
        self.dedu_toks = DeduceToks(ps, self, name='dedu_toks')
        self.out = tf.Dense(cfg.num_toks, name='out', activation=None)
        if cfg.beam_size:
            self.beam = Beam(ps, self, name='beam')

    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
    def call(self, inputs):
        enc, dec, ctx, tgt = inputs
        out = e_ms = d_ms = None
        if enc is not None:
            src, typ, mems = enc
            ctx, e_ms = self.encode(src, typ, mems)
        if dec is not None:
            src, typ, mems = dec
            ctx, d_ms = self.decode(src, typ, mems, ctx)
        if tf.learning_phase():
            out = self.dedu_loss([tgt, ctx])
        else:
            out = self.dedu_toks([tgt, ctx])
        return [out, e_ms, d_ms]

    def embed(self, x, typ=None):
        y = self.embed(x)
        if typ is not None and self.typ_emb:
            y = self.typ_emb([y, typ])
        if self.pos_emb:
            y = self.pos_emb(y)
        return y

    def encode(self, src, typ, mems):
        y = self.embed(src, typ)
        y, ms = self.enc_stack([y, mems])
        return y, ms

    def decode(self, src, typ, mems, ctx):
        y = self.embed(src, typ)
        y, ms = self.dec_stack([y, mems, ctx])
        return y, ms


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


class Encoder(Layer):
    def __init__(self, ps, owner, name, **kw):
        super().__init__(ps, **kw)
        self.refl = Attn(ps, owner, name=name + '_refl')
        self.ffnet = FFNet(ps, owner, name=name + '_ffnet')

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @tf.function
    def call(self, inputs):
        x, mem = inputs
        y = self.refl([x, mem])
        y = self.ffnet(y)
        return y


class Decoder(Encoder):
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
