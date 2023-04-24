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

# https://openai.com/blog/language-unsupervised/

import torch

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..prep.config.openai import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.tok_emb = qc.Embed(cfg.s_vocab, cfg.n_embed, **kw)
        self.pos_emb = qc.Embed(cfg.n_pos, cfg.n_embed, **kw)
        self.register_buffer("pos_ids", torch.arange(cfg.n_pos))
        self.drop = qc.Dropout(cfg.drop_embed, **kw)
        self.lays = qc.Stack([Layer(scale=True, **kw) for _ in range(cfg.n_lays)])

    def forward(self, x, head_m=None, mask=None, pos=None, typ=None, x_emb=None, **kw):
        cfg = self.cfg
        if x is None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x)
        if mask is not None:
            mask = self.get_mask(mask, s)
        head_m = self.get_head_m(head_m, cfg.n_lays)
        if pos is None:
            pos = self.pos_ids[None, : s[-1]]
        pos = self.pos_emb(pos)
        if typ is None:
            typ = 0
        else:
            typ = self.tok_emb(typ.view(-1, typ.size(-1)))
        y = self.drop(x_emb + pos + typ)
        attns = hiddens = ()
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            ys = lay(y, mask=mask, head_m=head_m[i])
            y = ys[0]
            attns += (ys[1],)
        y = y.view(*(s + (y.size(-1),)))
        hiddens += (y,)
        return qo.Base(y, attns, hiddens)


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.n_embed, cfg.n_labels, bias=False, **kw)

    forward = qf.forward_seq

    def post_proj(self, x):
        cfg = self.cfg
        b = (x.shape[:2] if x is not None else x_emb.shape[:2])[0]
        if cfg.PAD is None:
            n = -1
        else:
            assert b == 1
            n = -1 if x is None else torch.ne(x, cfg.PAD).sum(-1) - 1
        return x[torch.arange(b, device=self.device), n]


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.n_embed, cfg.s_vocab, bias=False, **kw)

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[..., :-1, :].contiguous()
            ls = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), ls.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


@dataclass
class Output(qc.Output):
    logits: tuple = None
    mc_logits: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    loss: tuple = None
    mc_loss: tuple = None


class DualHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        cfg.n_labels = 1
        self.model = Model(**kw)
        self.sum = qc.SeqSummary(**kw)
        self.proj = qc.Linear(cfg.n_embed, cfg.s_vocab, bias=False, **kw)

    def forward(self, x, mc_x=None, labels=None, mc_labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        mc_y = self.sum(ys[0], mc_x).squeeze(-1)
        loss, mc_loss = None, None
        if mc_labels is not None:
            mc_loss = nn.CrossEntropyLoss()(mc_y.view(-1, mc_y.size(-1)), mc_labels.view(-1))
        if labels is not None:
            sl = y[..., :-1, :].contiguous()
            ls = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), ls.view(-1))
        ys = (y, mc_y) + ys[1:] + (loss, mc_loss)
        return Output(*ys)


class Layer(qc.Module):
    hs = qc.Hypers({"d_model", "eps"})

    def __init__(self, scale=False, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.attn = Attention(scale, **kw)
        self.norm_attn = qc.LayerNorm(m, cfg.eps, **kw)
        self.proj = MLP(4 * m, **kw)
        self.norm = qc.LayerNorm(m, cfg.eps, **kw)

    def forward(self, x, mask, head_m, **kw):
        ys = self.attn(x, mask, head_m)
        y = self.norm_attn(x + ys[0])
        y = self.norm(y + self.proj(y))
        y = [y] + ys[1:]
        return y


class MLP(qc.Module):
    hs = qc.Hypers({"act", "drop"})

    def __init__(self, d_ff, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.conv = qc.Conv1D(d_ff, m, **kw)
        self.proj = qc.Conv1D(m, d_ff, **kw)
        self.act = qu.activation(cfg.act)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x):
        y = self.act(self.conv(x))
        y = self.drop(self.proj(y))
        return y


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "drop_attn", "drop", "n_heads", "n_pos"})

    def __init__(self, scale=False, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        cfg.scale = scale
        n, d = cfg.n_heads, cfg.d_model
        assert d % n == 0
        self.attn = qc.Conv1D(d * 3, d, **kw)
        self.proj = qc.Conv1D(d, d, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        p = cfg.n_pos
        self.register_buffer("bias", torch.tril(torch.ones(p, p)).view(1, 1, p, p))

    def forward(self, x, mask, head_m, **kw):
        cfg = self.cfg
        q, k, v = self.attn(x).split(cfg.d_model, dim=2)
        q = self.split_heads(q)
        k = self.split_heads(k, k=True)
        v = self.split_heads(v)
        ys = self.scores(q, k, v, mask, head_m)
        y = self.join_heads(ys[0])
        y = (self.drop(self.proj(y)),)
        return y + ys[1:]

    split_heads = qa.split_heads

    join_heads = qa.join_heads

    def scores(self, q, k, v, mask, head_m, **kw):
        cfg = self.cfg
        a = torch.matmul(q, k)
        if cfg.scale:
            a = a / (v.size(-1) ** 0.5)
        causal = self.bias[:, :, : a.size(-2), : a.size(-1)]
        a = a * causal + -1e4 * (1 - causal)
        if mask is not None:
            a = a + mask
        a = self.drop_attn(F.softmax(a, dim=-1))
        if head_m is not None:
            a = a * head_m
        return torch.matmul(a, v), a
