# Copyright 2023 Quantapix Authors. All Rights Reserved.
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

from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import attention as qa
from ..core import forward as qf
from ..core import output as qo
from ..core import utils as qu
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.albert import PreTrained

from . import bert

log = logging.get_logger(__name__)


class ForChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=1, **kw)

    forward = bert.ForChoice.forward


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(cfg.d_embed, **kw)

    forward = qf.forward_masked


class ForPreTraining(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Predictor(cfg.d_embed, **kw)
        self.next = Classifier(n_labels=2, **kw)

    forward = bert.ForPreTraining.forward


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class ForTokClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class Model(PreTrained):
    def __init__(self, add_pool=True, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.embs = Embeds(**kw)
        self.enc = Encoder(**kw)
        self.pool = Pool(**kw) if add_pool else None

    def forward(self, x, x_emb=None, mask=None, head_m=None, **kw):
        cfg = self.cfg
        if x is not None:
            assert x_emb is None
            s, d = x.size(), x.device
        else:
            s, d = x_emb.size()[:-1], x_emb.device
        if mask is None:
            mask = torch.ones(s, device=d)
        mask = self.get_mask(mask, s, d)
        head_m = self.get_head_m(head_m, cfg.n_lays)
        ys = self.embs(x, x_emb, **kw)
        ys = self.enc(ys, mask=mask, head_m=head_m, **kw)
        if self.pool is not None:
            ys += (self.pool(ys[0]),)
        return qo.WithPools(*ys)


class Encoder(qc.Module):
    hs = qc.Hypers({"d_embed", "d_model", "n_groups"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.proj = qc.Linear(cfg.d_embed, cfg.d_model, **kw)
        self.groups = qc.Stack([Group(**kw) for _ in range(cfg.n_groups)])

    def forward(self, x, head_m=None, **kw):
        cfg = self.cfg
        y = self.proj(x)
        attns = ()
        hiddens = ()
        hm = [None] * cfg.n_lays if head_m is None else head_m
        for i in range(cfg.n_lays):
            hiddens += (y,)
            n = int(cfg.n_lays / cfg.n_groups)
            g = int(i / (cfg.n_lays / cfg.n_groups))
            ys = self.groups[g](y, head_m=hm[g * n : (g + 1) * n], **kw)
            y = ys[0]
            attns += ys[1]
        hiddens += (y,)
        return qo.Base(y, attns, hiddens)


class Group(qc.Module):
    hs = qc.Hypers({"s_group"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([Layer(**kw) for _ in range(cfg.s_group)])

    def forward(self, x, head_m=None, **kw):
        y = x
        attns = ()
        hiddens = ()
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            ys = lay(y, head_m=head_m[i], **kw)
            y = ys[0]
            attns += (ys[1],)
        hiddens += (y,)
        return qo.Base(y, attns, hiddens)


class Layer(qc.Module):
    hs = qc.Hypers({"act", "d_ff", "d_model", "drop", "eps"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.attn = Attention(**kw)
        self.ffnet = MLP(**kw)
        self.norm = qc.LayerNorm(cfg.d_model, cfg.eps, **kw)

    def forward(self, x, **kw):
        ys = self.attn(x, **kw)
        y = self.ffnet(ys[0])
        y = self.norm(y + ys[0])
        return (y,) + ys[1:]


class Attention(qc.Module):
    hs = qc.Hypers(
        {"d_embed", "d_model", "n_heads", "n_pos"}, {"drop_attn": 0.0, "pos_type": "absolute"}
    )

    def __init__(self, pos_type=None, ps={}, hs=[], **kw):
        if pos_type is not None:
            kw.update(pos_type=pos_type)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0  # or cfg.d_embed is not None
        cfg.d_head = h = m // n
        cfg.scale = 1 / (h**0.5)
        self.query = qc.Linear(m, m, **kw)
        self.key = qc.Linear(m, m, **kw)
        self.value = qc.Linear(m, m, **kw)
        if cfg.pos_type == "relative_key" or cfg.pos_type == "relative_key_query":
            self.pos_emb = qc.Embed(2 * cfg.n_pos - 1, h, **kw)
        self.attn_drop = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(m, m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.norm = qc.LayerNorm(m, cfg.eps, **kw)

    split_heads = qa.split_heads

    def forward(self, x, mask=None, head_m=None, **kw):
        cfg = self.cfg
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        a = torch.matmul(q, k.transpose(-1, -2))
        a.mul_(cfg.scale)
        if mask is not None:
            a = a + mask
        t = cfg.pos_type
        if t == "relative_key" or t == "relative_key_query":
            n = x.size()[1]
            kw = dict(device=x.device, dtype=torch.long)
            left, right = torch.arange(n, **kw).view(-1, 1), torch.arange(n, **kw).view(1, -1)
            pos = self.pos_emb(left - right + self.n_pos - 1).to(dtype=q.dtype)
            if t == "relative_key":
                a += torch.einsum("bhld,lrd->bhlr", q, pos)
            elif t == "relative_key_query":
                a += torch.einsum("bhld,lrd->bhlr", q, pos) + torch.einsum("bhrd,lrd->bhlr", k, pos)
        a = self.attn_drop(F.softmax(a, dim=-1))
        if head_m is not None:
            a = a * head_m
        y = torch.matmul(a, v).transpose(2, 1).flatten(2)
        return self.norm(x + self.drop(self.proj(y))), a
