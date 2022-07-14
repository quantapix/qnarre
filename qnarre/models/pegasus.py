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
# https://arxiv.org/abs/1912.08777
# https://github.com/google-research/pegasus

import random
import torch

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import SinEmbed2
from ..prep.config.pegasus import PreTrained

from . import bart
from . import mbart


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.emb = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        self.enc = Encoder(self.emb, **kw)
        self.dec = Decoder(self.emb, **kw)

    def forward(
        self,
        x,
        dec_head_m=None,
        dec_m=None,
        mask=None,
        x_dec_emb=None,
        x_dec=None,
        y_enc=None,
        **kw,
    ):
        yo = self.get_y_opts(**kw)
        if y_enc is None:
            y_enc = self.enc(x, **kw, mask=mask, yo=yo)
        y = self.dec(
            x_dec,
            **kw,
            enc_m=mask,
            enc=y_enc[0],
            head_m=dec_head_m,
            mask=dec_m,
            x_emb=x_dec_emb,
            yo=yo,
        )
        ys = y + y_enc
        return qo.Seq2Seq(*ys) if yo.kw else ys


class ForCausal(PreTrained):
    def __init__(self, **kw):
        kw.update(is_dec=True, is_enc_dec=False)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Decoder(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, bias=False, **kw)

    forward = bart.ForCausal.forward


class ForCondGen(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.emb.num_embeddings)))
        self.proj = qc.Linear(cfg.d_model, self.model.emb.num_embeddings, bias=False, **kw)

    def forward(self, x, x_dec=None, labels=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if labels is not None:
            yo.cache = False
            if x_dec is None:
                x_dec = qu.shift_right(labels, cfg.PAD, cfg.dec_START)
        ys = self.model(x, x_dec=x_dec, **kw, yo=yo)
        y = self.proj(ys[0]) + self.final_logits_bias
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.s_vocab), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossSeq2Seq(*ys) if yo.kw else ys


class Encoder(PreTrained):
    hs = qc.Hypers(
        {"d_model", "n_enc_lays", "n_pos", "s_vocab", "scale"},
        {"drop": 0.0},
    )

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw) if tok_emb is None else tok_emb
        self.pos_emb = SinEmbed2(cfg.n_pos, m, **kw)
        self.lays = qc.Stack([mbart.EncLayer(**kw) for _ in range(cfg.n_enc_lays)])
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.grad_checkpoint = False

    def forward(self, x, head_m=None, mask=None, x_emb=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x is None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x) * cfg.scale
        y = x_emb + self.pos_emb(s)
        y = self.drop(y)
        attns = () if yo.attn else None
        hiddens = () if yo.hidden else None
        if mask is not None:
            mask = qu.expand_mask(mask, x_emb.dtype)
        assert head_m is None or (head_m.size()[0] == (len(self.lays)))
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y,)
            if self.training and (random.uniform(0, 1) < cfg.drop_enc):
                continue
            h = head_m[i] if head_m is not None else None
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, yo=yo)

                    return forward

                ys = checkpoint(create_forward(lay), head_m=h, mask=mask, **kw)
            else:
                ys = lay(y, head_m=h, mask=mask, **kw, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
        y = self.norm(y)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, hiddens)
        return qo.Base(*ys) if yo.kw else ys


class Decoder(PreTrained):
    hs = qc.Hypers(
        {"d_model", "n_dec_lays", "n_pos", "s_vocab", "scale"},
        {"drop": 0.0},
    )

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw) if tok_emb is None else tok_emb
        self.pos_emb = SinEmbed2(cfg.n_pos, m, **kw)
        self.lays = qc.Stack([mbart.DecLayer(**kw) for _ in range(cfg.n_dec_lays)])
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.grad_checkpoint = False

    def prep_dec_m(self, mask, shape, x_emb, c_len):
        y = None
        if shape[-1] > 1:
            y = qu.causal_mask(shape, x_emb.dtype, c_len=c_len).to(self.device)
        if mask is not None:
            m = qu.expand_mask(mask, x_emb.dtype, len=shape[-1])
            y = m if y is None else m + y
        return y

    def forward(
        self,
        x,
        cache=None,
        cross_m=None,
        enc_m=None,
        enc=None,
        head_m=None,
        mask=None,
        x_emb=None,
        **kw,
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x is None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x) * cfg.scale
        c_len = cache[0][0].shape[2] if cache is not None else 0
        y = x_emb + self.pos_emb(s, c_len)
        y = self.drop(y)
        attns = () if yo.attn else None
        caches = () if yo.cache else None
        crosses = () if (yo.attn and enc is not None) else None
        hiddens = () if yo.hidden else None
        mask = self.prep_dec_m(mask, s, x_emb, c_len)
        if enc is not None and enc_m is not None:
            enc_m = qu.expand_mask(enc_m, x_emb.dtype, len=s[-1])
        for m in [head_m, cross_m]:
            if m is not None:
                assert m.size()[0] == (len(self.lays))
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y,)
            if self.training and (random.uniform(0, 1) < cfg.drop_dec):
                continue
            h = head_m[i] if head_m is not None else None
            c = cross_m[i] if cross_m is not None else None
            kw.update(mask=mask, enc=enc, enc_m=enc_m, head_m=h, cross_m=c)
            c = cache[i] if cache is not None else None
            if self.grad_checkpoint and self.training:
                if yo.cache:
                    yo.cache = False

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c, yo=yo)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw)
            else:
                ys = lay(y, cache=c, **kw, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
                if enc is not None:
                    crosses += (ys[2],)
            if yo.cache:
                caches += (ys[-1],)
        y = self.norm(y)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys
