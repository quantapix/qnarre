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
# https://arxiv.org/abs/1810.04805
# https://github.com/google-research/bert

import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, FFNet, Masker, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, add_pool=True, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(cfg.d_model, **kw)
        self.enc = Encoder(**kw)
        self.pool = Pool(**kw) if add_pool else None

    def forward(
        self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, x_emb=None, **kw
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if not cfg.is_dec:
            yo.cache = False
        if x is None:
            s, d = x_emb.size()[:-1], x_emb.device
        else:
            assert x_emb is None
            s, d = x.size(), x.device
        c_len = cache[0][0].shape[2] if cache is not None else 0
        if mask is None:
            b, n = s
            mask = torch.ones(((b, n + c_len)), device=d)
        mask = self.get_mask(mask, s, d)
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        if not cfg.is_dec:
            yo.cache = False
        ys = self.embs(x, **kw, c_len=c_len, x_emb=x_emb, yo=yo)
        ys = self.enc(ys, **kw, cache=cache, enc_m=enc_m, enc=enc, head_m=head_m, mask=mask, yo=yo)
        if self.pool is not None:
            ys += (self.pool(ys[0]),)
        return qo.PoolsCrosses(*ys) if yo.kw else ys


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masker(**kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=1, **kw)

    def forward(self, x, x_emb=None, mask=None, typ=None, pos=None, labels=None, **kw):
        yo = self.get_y_opts(**kw)
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask, typ, pos = qu.view_2D(x, mask, typ, pos)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, x_emb=x_emb, mask=mask, typ=typ, pos=pos, **kw, yo=yo)
        y = self.proj(ys[1]).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys) if yo.kw else ys


class ForNextSentence(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.order = Classifier(n_labels=2, **kw)

    def forward(self, x, labels=None, **kw):
        yo = self.get_y_opts(**kw)
        ys = self.model(x, **kw, yo=yo)
        y = self.order(ys[1])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, 2), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys) if yo.kw else ys


class ForPreTraining(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(**kw)
        self.order = Classifier(n_labels=2, **kw)

    def forward(self, x, labels=None, order_label=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        ys = self.model(x, **kw, yo=yo)
        y = self.proj(ys[0])
        orders = self.order(ys[1])
        loss = None
        if labels is not None and order_label is not None:
            f = nn.CrossEntropyLoss()
            loss = f(y.view(-1, cfg.s_vocab), labels.view(-1)) + f(
                orders.view(-1, 2), order_label.view(-1)
            )
        ys = (y, orders) + ys[2:] + (loss,)
        return qo.LossSeq(*ys) if yo.kw else ys


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class Masked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masker(**kw)

    def forward(self, x, labels=None, **kw):
        yo = self.get_y_opts(**kw)
        if labels is not None:
            yo.cache = False
        ys = self.model(x, **kw, yo=yo)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[:, :-1, :].contiguous()
            ls = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, self.cfg.s_vocab), ls.view(-1))
        ys = (y,) + ys[2:] + (loss,)
        return qo.LossCrosses(*ys) if yo.kw else ys


class Encoder(qc.Module):
    hs = qc.Hypers({"add_cross", "n_lays"})

    def __init__(self, n_lays=None, ps={}, hs=[], **kw):
        if n_lays is not None:
            kw.update(n_lays=n_lays)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([Layer(**kw) for _ in range(cfg.n_lays)])
        self.grad_checkpoint = False

    def forward(self, x, head_m=None, cache=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        y = x
        attns = () if yo.attn else None
        caches = () if yo.cache else None
        crosses = () if yo.attn and cfg.add_cross else None
        hiddens = () if yo.hidden else None
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y,)
            h = head_m[i] if head_m is not None else None
            c = cache[i] if cache is not None else None
            if cfg.grad_checkpoint and self.training:
                if yo.cache:
                    yo.cache = False

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c, yo=yo)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw, mask=h)
            else:
                ys = lay(y, **kw, cache=c, mask=h, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
                if cfg.add_cross:
                    crosses += (ys[2],)
            if yo.cache:
                caches += (ys[-1],)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, cache, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys


class Layer(qc.Module):
    hs = qc.Hypers({"act", "add_cross"}, {"is_dec": False})

    def __init__(self, add_cross=None, ps={}, hs=[], **kw):
        if add_cross is not None:
            kw.update(add_cross=add_cross)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.attn = Attention(**kw)
        if cfg.add_cross:
            assert cfg.is_dec
            self.cross = Attention(**kw)
        self.proj = FFNet(**kw)

    def forward(self, x, cache=None, enc=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, y_cache=True, yo=None)
        c = cache[:2] if cache is not None else None
        y, a, kv = self.attn(x, cache=c, **kw)
        a2 = None
        if cfg.is_dec and enc is not None:
            c = cache[-2:] if cache is not None else None
            y, a2, kv2 = self.cross(y, cache=c, enc=enc, **kw)
            kv = kv + kv2
        y = (self.proj(y),)
        if yo.attn:
            y += (a, a2)
        if yo.cache:
            y += (kv,)
        return y


class Attention(qc.Module):
    hs = qc.Hypers(
        {"d_model", "drop_proj", "drop", "n_heads", "n_pos", "eps"},
        {"drop_attn": 0.0, "pos_type": "absolute"},
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
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        if cfg.pos_type == "relative_key" or cfg.pos_type == "relative_key_query":
            self.pos_emb = qc.Embed(2 * cfg.n_pos - 1, h, **kw)
        self.proj = qc.Linear(m, m, **kw)
        p = cfg.drop_proj if cfg.drop_proj is not None else cfg.drop
        self.drop = qc.Dropout(p, **kw)
        self.norm = qc.LayerNorm(m, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        q = self.split_heads(self.query(x))
        if enc is None:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
            if cache is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
        else:
            mask = enc_m
            if cache is None:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
            else:
                k = cache[0]
                v = cache[1]
        a = torch.matmul(q, k.transpose(-1, -2))
        t = cfg.pos_type
        if t == "relative_key" or t == "relative_key_query":
            n = x.size()[1]
            kw = dict(device=x.device, dtype=torch.long)
            left, right = torch.arange(n, **kw).view(-1, 1), torch.arange(n, **kw).view(1, -1)
            pos = self.pos_emb((left - right) + cfg.n_pos - 1).to(dtype=q.dtype)
            if t == "relative_key":
                a += torch.einsum("bhld,lrd->bhlr", q, pos)
            elif t == "relative_key_query":
                a += torch.einsum("bhld,lrd->bhlr", q, pos) + torch.einsum("bhrd,lrd->bhlr", k, pos)
        a.mul_(cfg.scale)
        if mask is not None:
            a += mask
        a = self.drop_attn(F.softmax(a, dim=-1))
        if head_m is not None:
            a *= head_m
        y = torch.matmul(a, v).permute(0, 2, 1, 3).contiguous()
        y = y.view(y.size()[:-2] + (cfg.n_heads * cfg.d_head,))
        y = (self.norm(x + self.drop(self.proj(y))),)
        if yo.attn:
            y += (a,)
        if yo.cache:
            y += ((k, v),)
        return y
