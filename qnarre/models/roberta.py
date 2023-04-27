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
# https://arxiv.org/abs/1907.11692

import torch

from torch import nn
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core.embed import Embeds
from ..core.mlp import MLP, Classifier, Predictor
from ..prep.config.roberta import PreTrained

from . import bert

log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, add_pool=True, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(cfg.d_model, **kw)
        self.enc = Encoder(**kw)
        self.pool = qu.Pool(**kw) if add_pool else None

    def forward(
        self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, x_emb=None, **kw
    ):
        cfg = self.cfg
        if x is None:
            s, d = x_emb.size()[:-1], x_emb.device
        else:
            assert x_emb is None
            s, d = x.size(), x.device
        c_len = cache[0][0].shape[2] if cache is not None else 0
        if mask is None:
            b, n = s
            mask = torch.ones(((b, n + c_len)), device=d)
        xm = self.get_mask(mask, s, d)
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        ys = self.embs(x, **kw, c_len=c_len, x_emb=x_emb)
        ys = self.enc(ys, **kw, cache=cache, enc_m=enc_m, enc=enc, head_m=head_m, mask=xm)
        pools = self.pool(ys[0]) if self.pool is not None else None
        ys += (pools,)
        return qo.PoolsCrosses(*ys)


class ForCausal(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = bert.LMHead(**kw)

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[:, :-1, :].contiguous()
            ls = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, cfg.s_vocab), ls.view(-1))
        ys = (y,) + ys[2:] + (loss,)
        return qo.LossCrosses(*ys)


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(**kw)

    forward = qf.forward_masked


class ForChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=1, **kw)

    def forward(self, x, typ=None, mask=None, labels=None, pos=None, x_emb=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask, typ, pos = qu.view_2D(x, mask, typ, pos)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, pos=pos, typ=typ, mask=mask, x_emb=x_emb, **kw)
        y = self.proj(ys[1]).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[2:] + (loss,)
        return qo.WithLoss(*ys)


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
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class ForTokClass(PreTrained):
    def __init__(self, *kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


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
        y = x
        attns = caches = crosses = hiddens = ()
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            h = head_m[i] if head_m is not None else None
            c = cache[i] if cache is not None else None
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c)

                    return forward

                ys = checkpoint(create_forward(lay), y, mask=h, **kw)
            else:
                ys = lay(y, mask=h, cache=c, **kw)
            y = ys[0]
            attns += (ys[1],)
            if cfg.add_cross:
                crosses += (ys[2],)
            caches += (ys[-1],)
        hiddens += (y,)
        return qo.CachesCrosses(y, attns, caches, crosses, hiddens)


class Layer(qc.Module):
    hs = qc.Hypers({"add_cross", "act"}, {"is_dec": False})

    def __init__(self, add_cross=None, ps={}, hs=[], **kw):
        if add_cross is not None:
            kw.update(add_cross=add_cross)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.attn = bert.Attention(**kw)
        if cfg.add_cross:
            assert cfg.is_dec
            self.cross = bert.Attention(pos_type="absolute", **kw)
        self.proj = MLP(cfg.act, cfg.drop, cfg.eps, **kw)

    def forward(self, x, cache=None, enc=None, **kw):
        cfg = self.cfg
        c = cache[:2] if cache is not None else None
        y, a, kv = self.attn(x, cache=c, **kw)
        a2 = None
        if cfg.is_dec and enc is not None:
            c = cache[-2:] if cache is not None else None
            y, a2, kv2 = self.cross(y, cache=c, enc=enc, **kw)
            kv = kv + kv2
        return self.proj(y), a, a2, kv
