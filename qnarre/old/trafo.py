# Copyright 2022 Quantapix Authors. All Rights Reserved.
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

import torch

from qnarre.core.attention import Attend
from qnarre.core.base import Hypers, Module, Linear
from qnarre.core.ffnet import FFNet
from qnarre.core.deduce import Deduce, Search
from qnarre.core.norm import PreProc, PostProc
from qnarre.core.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming


def adapter(ps, feats, x):
    d = torch.parse_example(x, feats)
    img = torch.to_dense(d["flt_img"])
    # img = torch.cast(d['int_img'], torch.float32) / 255.
    lbl = d["int_lbl"]
    return img, lbl


def model(ps):
    src = torch.Input(shape=(ps.len_src,), dtype="int32")
    typ = torch.Input(shape=(ps.len_src,), dtype="int32")
    hint = torch.Input(shape=(ps.len_tgt,), dtype="int32")
    tgt = torch.Input(shape=(ps.len_tgt,), dtype="int32")
    ins = [src, typ, hint, tgt]
    outs = [Trafo(ps)(ins)]
    m = torch.Model(name="TrafoModel", inputs=ins, outputs=outs)
    return m


class Trafo(Module):
    hs = Hypers(
        [
            "beam_size",
            "drop_hidden",
            "len_src",
            "len_tgt",
            "num_toks",
            "pos_type",
            "n_typ",
        ],
        {},
    )

    typ_embed = pos_embed = enc_stack = dec_stack = pos_x_b = pos_p_b = None

    def __init__(self, dim_out=None, hs=[], **kw):
        if dim_out is not None:
            kw.update(dim_out=dim_out)
        super().__init__([self.hs] + hs, **kw)
        cfg = self.cfg
        kw.update(hs=hs)
        self.tok_embed = TokEmbed(**kw)
        if cfg.n_typ:
            self.typ_embed = TypEmbed(**kw)
        if cfg.pos_type == "embed":
            self.pos_embed = PosEmbed(**kw)
        elif cfg.pos_type == "timing":
            self.pos_embed = PosTiming(**kw)
        else:
            assert cfg.pos_type == "relative"
        self.pre = PreProc(**kw)
        self.post = PostProc(**kw)
        self.enc_stack = EncStack(self, **kw)
        self.dec_stack = DecStack(self, **kw)
        self.deduce = Deduce(self, **kw)
        self.search = Search(self, **kw)
        self.out = Linear(cfg.num_toks, **kw)

    def forward(self, x, training=None):
        src, typ, hint, tgt = x
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
        y = self.tok_embed(x)
        if self.typ_embed and typ is not None:
            y = self.typ_embed([y, typ])
        if self.pos_embed:
            y = self.pos_embed(y)
        return y


class Stack(Module):
    def __init__(self, owner, ps=None, **kw):
        super().__init__(ps, **kw)
        self.pre = owner.pre
        self.post = owner.post


class EncStack(Stack):
    hs = Hypers(["n_encoders"], {})

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, owner, **kw)
        cfg = self.cfg
        n = cfg.n_encoders
        self.encs = [Encoder(ps, owner, f"enc_{i}") for i in range(n)]

    def forward(self, x):
        x = x[0]
        y = self.pre([x, x])
        for e in self.encs:
            y = e([y])
        y = self.post([x, y])
        return y


class DecStack(Stack):
    hs = Hypers(["num_dec_lays"], {})

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, owner, **kw)
        cfg = self.cfg
        n = cfg.num_dec_lays
        self.decs = [Decoder(ps, owner, f"dec_{i}") for i in range(n)]

    def forward(self, x):
        x, ctx = x
        """
        cfg = self.cfg
        if ps.causal_refl:
            if ps.prepend_mode == 'prepend_inputs_full_attention':
                y = torch.cumsum(torch.cumsum(rb, axis=1), axis=1)
                y2 = torch.expand_dims(y, axis=1)
                y = torch.greater(y2, torch.expand_dims(y, axis=2))
                b = torch.expand_dims(torch.cast(y, torch.floatx()) * -1e9, axis=1)
            else:
                ln = torch.int_shape(x)[1]
                sh = (1, 1, ln, ln)
                b = U.ones_band_part(ln, ln, -1, 0, out_shape=sh)
                b = -1e9 * (1.0 - b)
        """
        y = self.pre([x, x])
        for d in self.decs:
            y = d([y, ctx])
        y = self.post([x, y])
        return y


class Encoder(Module):
    hs = Hypers(
        ["len_mem"],
        {},
    )
    mem = None

    def __init__(self, owner, ps=None, name="enc", **kw):
        super().__init__(ps, name=name, **kw)
        self.refl = Attend(owner, ps, name=name + "_refl")
        self.ffnet = FFNet(owner, ps, name=name + "_ffnet")
        mlen = self.cfg.len_mem
        if mlen:
            s = input_shape[0]
            s = s[:1] + (mlen,) + s[2:]
            self.mem = self.add_resource(self.name + "_mem", s)

    def forward(self, x):
        x = x[0]
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
            self.mem.assign(torch.concat([m, x], axis=1)[:, -i:])
        return y


class Decoder(Encoder):
    def __init__(self, owner, ps=None, name="dec", **kw):
        super().__init__(owner, ps, name, **kw)
        self.attn = Attend(owner, ps, name=name + "_attn")

    def forward(self, x):
        x, ctx = x
        y = self.reflect(x)
        if ctx is not None:
            y = self.attn([y, ctx])
        y = self.ffnet(y)
        return y
