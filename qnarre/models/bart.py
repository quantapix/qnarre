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
# https://arxiv.org/abs/1910.13461
# https://github.com/pytorch/fairseq/tree/main/examples/bart

import math
import random
import torch

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import PosEmbed
from ..core.mlp import Classifier
from ..prep.config.bart import PreTrained


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
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x_dec is None and x_dec_emb is None:
            assert x is not None
            x_dec = qu.shift_right(x, cfg.PAD, cfg.dec_START)
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

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        ys = self.model(x, **kw, yo=yo)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.s_vocab), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCrosses(*ys) if yo.kw else ys


class ForCondGen(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        n = self.model.emb.cfg.n_embed
        self.proj = qc.Linear(cfg.d_model, n, bias=False, **kw)
        self.register_buffer("final_logits_bias", torch.zeros((1, n)))

    def forward(self, x, labels=None, x_dec_emb=None, x_dec=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if labels is not None:
            yo.cache = False
            if x_dec is None and x_dec_emb is None:
                x_dec = qu.shift_right(labels, cfg.PAD, cfg.dec_START)
        ys = self.model(x, x_dec=x_dec, x_dec_emb=x_dec_emb, **kw, yo=yo)
        y = self.proj(ys[0]) + self.final_logits_bias
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.s_vocab), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossSeq2Seq(*ys) if yo.kw else ys


class ForQA(PreTrained):
    def __init__(self, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, **kw)

    forward = qf.forward_seq

    def pre_proj(self, x, ys):
        y = ys[0]
        eos_m = x.eq(self.cfg.EOS)
        assert len(torch.unique_consecutive(eos_m.sum(1))) <= 1
        y = y[eos_m, :].view(y.size(0), -1, y.size(-1))
        return y[:, -1, :]


class Encoder(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw) if tok_emb is None else tok_emb
        self.pos_emb = PosEmbed(cfg.n_pos, m, **kw)
        self.lays = qc.Stack([EncLayer(**kw) for _ in range(cfg.n_enc_lays)])
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
        y = self.drop(self.norm(y))
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

                ys = checkpoint(create_forward(lay), y, head_m=h, mask=mask, **kw)
            else:
                ys = lay(y, head_m=h, mask=mask, **kw, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, hiddens)
        return qo.Base(*ys) if yo.kw else ys


class Decoder(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw) if tok_emb is None else tok_emb
        self.pos_emb = PosEmbed(cfg.n_pos, m, **kw)
        self.lays = qc.Stack([DecLayer(**kw) for _ in range(cfg.n_dec_lays)])
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
        y = self.drop(self.norm(y))
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
            kw.update(cross_m=c, enc_m=enc_m, enc=enc, head_m=h, mask=mask)
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
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys


class EncLayer(qc.Module):
    hs = qc.Hypers(
        {"act", "d_enc_ff", "d_model", "drop_act", "n_enc_heads", "eps"},
        {"drop": 0.0, "is_dec": False},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_enc_heads, **kw)
        self.norm_refl = qc.LayerNorm(m, **kw)
        self.act = qu.activation(cfg.act)
        self.drop_act = qc.Dropout(cfg.drop_act, **kw)
        self.ff = qc.Linear(m, cfg.d_enc_ff, **kw)
        self.proj = qc.Linear(cfg.d_enc_ff, m, **kw)
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, **kw):
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, yo=None)
        y, a, _ = self.refl(x, **kw)
        y = self.norm_refl(x + self.drop(y))
        x = y
        y = self.drop_act(self.act(self.ff(y)))
        y = self.drop(self.proj(y))
        y = self.norm(x + y)
        if y.dtype == torch.float16 and (torch.isinf(y).any() or torch.isnan(y).any()):
            clamp = torch.finfo(y.dtype).max - 1000
            y = torch.clamp(y, min=-clamp, max=clamp)
        y = (y,)
        if yo.attn:
            y += (a,)
        return y


class DecLayer(qc.Module):
    hs = qc.Hypers(
        {"act", "d_dec_ff", "d_model", "drop_act", "n_dec_heads", "eps"},
        {"drop": 0.0, "is_dec": False},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_dec_heads, is_dec=True, **kw)
        self.norm_refl = qc.LayerNorm(m, **kw)
        self.act = qu.activation(cfg.act)
        self.drop_act = qc.Dropout(cfg.drop_act, **kw)
        self.attn = Attention(n_heads=cfg.n_dec_heads, is_dec=True, **kw)
        self.norm_attn = qc.LayerNorm(m, **kw)
        self.ff = qc.Linear(m, cfg.d_dec_ff, **kw)
        self.proj = qc.Linear(cfg.d_dec_ff, m, **kw)
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, cache=None, cross_m=None, enc_m=None, enc=None, **kw):
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, y_cache=True, yo=None)
        c = cache[:2] if cache is not None else None
        y, a, kv = self.refl(x, cache=c, **kw)
        y = self.norm_refl(x + self.drop(y))
        a2 = None
        if enc is not None:
            x = y
            c = cache[-2:] if cache is not None else None
            y, a2, kv2 = self.attn(y, cache=c, enc=enc, head_m=cross_m, mask=enc_m, **kw)
            y = self.norm_attn(x + self.drop(y))
            kv = kv + kv2
        x = y
        y = self.drop_act(self.act(self.ff(y)))
        y = self.drop(self.proj(y))
        y = self.norm(x + y)
        y = (y,)
        if yo.attn:
            y += (a, a2)
        if yo.cache:
            y += (kv,)
        return y


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "scale"}, {"drop_attn": 0.0})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        n, d = cfg.n_heads, cfg.d_model
        cfg.s_head = int(d / n)
        cfg.scale = cfg.s_head**-0.5
        self.query = qc.Linear(d, d, **kw)
        self.key = qc.Linear(d, d, **kw)
        self.value = qc.Linear(d, d, **kw)
        self.drop = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(d, d, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        q = self.split_heads(self.query(x) * cfg.scale)
        if enc is None:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
            if cache is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
        else:
            if cache is None:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
            else:
                k, v = cache
        n = cfg.n_heads
        b, tgt, _ = x.size()
        s = (b * n, -1, cfg.s_head)
        y = torch.bmm(q.view(s), k.view(s).transpose(1, 2))
        src = k.view(s).size(1)
        assert y.size() == (b * n, tgt, src)
        if mask is not None:
            assert mask.size() == (b, 1, tgt, src)
            y = y.view(b, n, tgt, src) + mask
            y = y.view(b * n, tgt, src)
        y = F.softmax(y, dim=-1)
        if head_m is not None:
            assert head_m.size() == (n,)
            y = head_m.view(1, -1, 1, 1) * y.view(b, n, tgt, src)
            y = y.view(b * n, tgt, src)
        if yo.attn:
            a = y.view(b, n, tgt, src)
            y = a.view(b * n, tgt, src)
        y = torch.bmm(self.drop(y), v.view(s))
        assert y.size() == (b * n, tgt, cfg.s_head)
        y = y.view(b, n, tgt, cfg.s_head)
        y = y.transpose(1, 2).reshape(b, tgt, cfg.d_model)
        y = self.proj(y)
        y = (y,)
        if yo.attn:
            y += (a,)
        if yo.cache:
            y += ((k, v),)
        return y
