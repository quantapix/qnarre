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
# https://openreview.net/pdf?id=r1xMH1BtvB
# https://github.com/google-research/electra

import torch
import torch.utils.checkpoint

from torch import nn
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core.embed import Embeds
from ..core.mlp import Classifier, Masked
from ..prep.config.electra import PreTrained

from . import bert


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(**kw)
        if cfg.d_embed != cfg.d_model:
            self.proj = qc.Linear(cfg.d_embed, cfg.d_model, **kw)
        self.enc = Encoder(**kw)

    def forward(
        self, x, x_emb=None, mask=None, head_m=None, enc=None, enc_m=None, cache=None, **kw
    ):
        cfg = self.cfg
        if x is not None:
            assert x_emb is None
            s, d = x.size(), x.device
        else:
            s, d = x_emb.size()[:-1], x_emb.device
        c_len = cache[0][0].shape[2] if cache is not None else 0
        if mask is None:
            mask = torch.ones(s, device=d)
        mask = self.get_mask(mask, s, d)
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        ys = self.embs(x, x_emb=x_emb, c_len=c_len, **kw)
        if hasattr(self, "proj"):
            ys = self.proj(ys)
        ys = self.enc(ys, mask=mask, head_m=head_m, enc=enc, enc_m=enc_m, cache=cache, **kw)
        return ys


class Generator(qc.Module):
    hs = qc.Hypers({"d_embed", "d_model", "drop", "eps", "s_vocab"}, {"act": "gelu"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lin = qc.Linear(cfg.d_model, cfg.d_embed, **kw)
        self.act = qu.activation(cfg.act)
        self.norm = qc.LayerNorm(cfg.d_embed, cfg.eps, **kw)
        self.proj = qc.Linear(cfg.d_embed, cfg.s_vocab, **kw)

    def forward(self, x):
        y = self.lin(x)
        y = self.act(y)
        y = self.norm(y)
        y = self.proj(y)
        return y


class ForCausal(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Generator(**kw)
        self.init_weights()

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[:, :-1, :].contiguous()
            ls = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, cfg.s_vocab), ls.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCrosses(*ys)


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masked(cfg.d_embed, **kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.seqs = qc.SeqSummary(**kw)
        self.proj = qc.Linear(cfg.d_model, 1, **kw)

    def forward(self, x, x_emb=None, mask=None, typ=None, pos=None, labels=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask, typ, pos = qu.view_2D(x, mask, typ, pos)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, x_emb, mask=mask, typ=typ, pos=pos, **kw)
        y = self.proj(self.seqs(ys[0])).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForPreTraining(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, cfg.act, n_labels=1, drop_proj=0.0, **kw)

    def forward(self, x, mask=None, labels=None, **kw):
        ys = self.model(x, mask=mask, **kw)
        y = self.proj(ys[0]).squeeze(-1)
        loss = None
        if labels is not None:
            f = nn.BCEWithLogitsLoss()
            if mask is not None:
                a = mask.view(-1, ys[0].shape[1]) == 1
                loss = f(y.view(-1, ys[0].shape[1])[a], labels[a].float())
            else:
                loss = f(y.view(-1, ys[0].shape[1]), labels.float())
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, "gelu", **kw)

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
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class Encoder(qc.Module):
    hs = qc.Hypers({"add_cross", "n_lays"})

    def __init__(self, n_lays=None, ps={}, hs=[], **kw):
        if n_lays is not None:
            kw.update(n_lays=n_lays)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([bert.Layer(**kw) for _ in range(cfg.n_lays)])
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

                ys = torch.utils.checkpoint.checkpoint(create_forward(lay), y, head_m=h, **kw)
            else:
                ys = lay(y, head_m=h, cache=c, **kw)
            y = ys[0]
            attns += (ys[1],)
            if cfg.add_cross:
                crosses += (ys[2],)
            caches += (ys[-1],)
        hiddens += (y,)
        return qo.CachesCrosses(y, attns, caches, crosses, hiddens)
