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
# https://arxiv.org/abs/1910.01108

import torch
import deepspeed

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import FFNet, Classifier, Masker
from ..prep.config.distilbert import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(cfg.d_model, **kw)
        self.enc = Encoder(**kw)

    def forward(self, x, x_emb=None, mask=None, head_m=None, **kw):
        cfg = self.cfg
        if x is not None:
            assert x_emb is None
            s, d = x.size(), x.device
        else:
            s, d = x_emb.size()[:-1], x_emb.device
        if mask is None:
            mask = torch.ones(s, device=d)
        head_m = self.get_head_m(head_m, cfg.n_lays)
        y = self.embs(x, x_emb, **kw)
        y = self.enc(y, mask=mask, head_m=head_m)
        return y


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masker(cfg.d_model, eps=1e-12, **kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, "relu", n_labels=1, drop_proj=cfg.drop_seq, **kw)

    def forward(self, x, x_emb=None, mask=None, labels=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask = qu.view_2D(x, mask)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, x_emb, mask=mask, **kw)
        y = self.proj(ys[0][:, 0]).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, "relu", drop_proj=cfg.drop_seq, **kw)

    forward = qf.forward_seq


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=2, drop_proj=cfg.drop_qa, **kw)

    forward = qf.forward_qa


class Encoder(qc.Module):
    hs = qc.Hypers({"n_lays"})

    def __init__(self, n_lays=None, ps={}, hs=[], **kw):
        if n_lays is not None:
            kw.update(n_lays=n_lays)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([Layer(**kw) for _ in range(cfg.n_lays)])

    def forward(self, x, head_m=None, **kw):
        y = x
        attns = hiddens = ()
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            ys = lay(y, head_m=head_m[i], **kw)
            y = ys[0]
            attns += (ys[1],)
        hiddens += (y,)
        return qo.Base(y, attns, hiddens)


class Layer(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        assert cfg.d_model % cfg.n_heads == 0
        self.refl = Attention(**kw)
        assert cfg.activation in ["relu", "gelu"]
        self.ffnet = FFNet(act=cfg.activation, drop=cfg.drop, **kw)
        self.norm = qc.LayerNorm(cfg.d_model, 1e-12)

    def forward(self, x, **kw):
        y, a = self.refl(x, **kw)
        y = self.norm(self.ffnet(y) + y)
        return y, a


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads"}, {"drop_attn": 0.0, "eps": 1e-12})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0
        cfg.d_head = h = m // n
        cfg.scale = 1 / (h**0.5)
        self.query = qc.Linear(m, m, **kw)
        self.key = qc.Linear(m, m, **kw)
        self.value = qc.Linear(m, m, **kw)
        self.proj = qc.Linear(m, m, **kw)
        self.drop = qc.Dropout(cfg.drop_attn, **kw)
        self.norm = qc.LayerNorm(cfg.d_model, **kw)

    split_heads = qa.split_heads

    def forward(self, x, head_m=None, mask=None, **kw):
        cfg = self.cfg
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        q.mul_(cfg.scale)
        a = torch.matmul(q, k.transpose(2, 3))
        b = x.size()[0]
        mask = (mask == 0).view((b, 1, 1, x.size(1))).expand_as(a)
        a = a.masked_fill(mask, -float("inf"))
        a = self.drop(F.softmax(a, dim=-1))
        if head_m is not None:
            a *= head_m
        y = torch.matmul(a, v).transpose(1, 2).contiguous()
        y = y.view(b, -1, cfg.n_heads * cfg.d_head)
        y = self.norm(x + self.proj(y))
        return y, a
