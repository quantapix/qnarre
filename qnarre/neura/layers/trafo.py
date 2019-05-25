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
from qnarre.neura.layers.deduce import Deduce, Search
from qnarre.neura.layers.norm import PreProc, PostProc
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming


def adapter(ps, feats, x):
    d = tf.parse_example(x, feats)
    img = tf.to_dense(d['flt_img'])
    # img = tf.cast(d['int_img'], tf.float32) / 255.
    lbl = d['int_lbl']
    return img, lbl


def model(ps):
    src = tf.Input(shape=(ps.len_src, ), dtype='int32')
    typ = tf.Input(shape=(ps.len_src, ), dtype='int32')
    hint = tf.Input(shape=(ps.len_tgt, ), dtype='int32')
    tgt = tf.Input(shape=(ps.len_tgt, ), dtype='int32')
    ins = [src, typ, hint, tgt]
    outs = [Trafo(ps)(ins)]
    m = tf.Model(name='TrafoModel', inputs=ins, outputs=outs)
    return m


class Trafo(Layer):
    typ_emb = pos_emb = enc_stack = dec_stack = pos_x_b = pos_p_b = None

    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'beam_size',
                'drop_hidden',
                'len_src',
                'len_tgt',
                'num_toks',
                'pos_type',
                'tok_types',
            ))

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        self.tok_emb = TokEmbed(ps, name='tok_emb')
        if cfg.tok_types:
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
        self.deduce = Deduce(ps, self, name='deduce')
        self.search = Search(ps, self, name='search')
        self.out = tf.Dense(cfg.num_toks, name='out', activation=None)

    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[3]

    @tf.function
    def call(self, inputs, training=None):
        src, typ, hint, tgt = inputs
        ctx = None
        if src is not None:
            y = self.embed(src, typ)
            ctx = self.enc_stack([y])
        if hint is not None:
            y = self.embed(hint)
            ctx = self.dec_stack([y, ctx])
        if training is not None:
            out = self.deduce([ctx, tgt])
        else:
            # out = self.search([tgt, ctx])
            pass
        return out

    def embed(self, x, typ=None):
        y = self.tok_emb(x)
        if self.typ_emb and typ is not None:
            y = self.typ_emb([y, typ])
        if self.pos_emb:
            y = self.pos_emb(y)
        return y


class Stack(Layer):
    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        self.pre = owner.pre
        self.post = owner.post

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class EncStack(Stack):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items(
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
        x = inputs[0]
        y = self.pre([x, x])
        for i, e in enumerate(self.encs):
            y = e([y])
        y = self.post([x, y])
        return y


class DecStack(Stack):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items(
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
        x, ctx = inputs
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
        y = self.pre([x, x])
        for i, d in enumerate(self.decs):
            y = d([y, ctx])
        y = self.post([x, y])
        return y


class Encoder(Layer):
    mem = None

    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items('len_mem', ))

    def __init__(self, ps, owner, name, **kw):
        super().__init__(ps, name=name, **kw)
        self.refl = Attn(ps, owner, name=name + '_refl')
        self.ffnet = FFNet(ps, owner, name=name + '_ffnet')

    def build(self, input_shape):
        mlen = self.cfg.len_mem
        if mlen:
            s = input_shape[0]
            s = s[:1] + (mlen, ) + s[2:]
            self.mem = self.add_resource(self.name + '_mem', s)
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @tf.function
    def call(self, inputs):
        x = inputs[0]
        y = self.reflect(x)
        y = self.ffnet(y)
        return y

    def reflect(self, x):
        m = self.mem
        if m is None:
            y = self.refl([x])
        else:
            y = self.refl([x, m])
            i = self.cfg.len_mem
            self.mem.assign(tf.concat([m, x], axis=1)[:, -i:, ])
        return y


class Decoder(Encoder):
    def __init__(self, ps, owner, name, **kw):
        super().__init__(ps, owner, name, **kw)
        self.attn = Attn(ps, owner, name=name + '_attn')

    @tf.function
    def call(self, inputs):
        x, ctx = inputs
        y = self.reflect(x)
        if ctx is not None:
            y = self.attn([y, ctx])
        y = self.ffnet(y)
        return y
