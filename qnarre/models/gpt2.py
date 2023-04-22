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
# https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# https://openai.com/blog/better-language-models/

from dataclasses import dataclass
import torch

from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.ffnet import Classifier
from ..prep.cfg.gpt2 import PreTrained

from . import openai

log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.tok_emb = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        self.pos_emb = qc.Embed(cfg.n_pos, cfg.d_model, **kw)
        self.lays = qc.Stack([Layer(lay_i=i, **kw) for i in range(cfg.n_lays)])
        self.norm = qc.LayerNorm(cfg.d_model, cfg.eps, **kw)
        self.drop = qc.Dropout(cfg.drop_embed, **kw)

    def forward(
        self,
        x,
        cache=None,
        enc_m=None,
        enc=None,
        head_m=None,
        mask=None,
        pos=None,
        typ=None,
        x_emb=None,
        **kw,
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x is None:
            s, d = x_emb.size()[:-1], x_emb.device
        else:
            assert x_emb is None
            s, d = x.size(), x.device
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x)
        if cache is None:
            c_len = 0
            cache = tuple([None] * len(self.lays))
        else:
            c_len = cache[0][0].size(-2)
        if mask is not None:
            mask = self.get_mask(mask.view(s[0], -1), s, d)
        if pos is not None:
            pos = pos.view(-1, s[-1])
        else:
            pos = (
                torch.arange(c_len, s[-1] + c_len, dtype=torch.long, device=d)
                .unsqueeze(0)
                .view(-1, s[-1])
            )
        if typ is not None:
            typ = typ.view(-1, s[-1])
        if cfg.add_cross and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        y = x_emb + self.pos_emb(pos)
        if typ is not None:
            y = y + self.tok_emb(typ)
        y = self.drop(y)
        attns = () if yo.attn else None
        caches = () if yo.cache else None
        crosses = () if yo.attn and cfg.add_cross else None
        hiddens = () if yo.hidden else None
        for i, (lay, c) in enumerate(zip(self.lays, cache)):
            if yo.hidden:
                hiddens += (y,)
            kw.update(enc_m=enc_m, enc=enc, head_m=head_m[i], mask=mask)
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
                attns += (ys[2 if yo.cache else 1],)
                if cfg.add_cross:
                    crosses += (ys[3 if yo.cache else 2],)
            if yo.cache:
                caches += (ys[1],)
        y = self.norm(y).view(s + (y.size(-1),))
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys


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


class ForTokClassifier(PreTrained):
    def __init__(self, drop_proj=0.1, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(drop_proj=drop_proj, **kw)

    forward = qf.forward_tok


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.n_embed, cfg.s_vocab, bias=False, **kw)

    def forward(self, x, labels=None, **kw):
        yo = self.get_y_opts(**kw)
        ys = self.model(x, **kw, yo=yo)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[..., :-1, :].contiguous()
            ls = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), ls.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCrosses(*ys) if yo.kw else ys


@dataclass
class Output(qc.Output):
    logits: tuple = None
    mc_logits: tuple = None
    attns: tuple = None
    caches: tuple = None
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

    def forward(self, x, mc_token_ids=None, labels=None, mc_labels=None, **kw):
        yo = self.get_y_opts(**kw)
        ys = self.model(x, **kw, yo=yo)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[..., :-1, :].contiguous()
            ls = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), ls.view(-1))
        mc_y = self.sum(ys[0], mc_token_ids).squeeze(-1)
        mc_loss = None
        if mc_labels is not None:
            mc_loss = nn.CrossEntropyLoss()(mc_y.view(-1, mc_y.size(-1)), mc_labels.view(-1))
        ys = (y, mc_y) + ys[1:] + (loss, mc_loss)
        return Output(*ys) if yo.kw else ys


class Layer(qc.Module):
    def __init__(self, lay_i, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.attn = Attention(lay_i=lay_i, **kw)
        self.norm_attn = qc.LayerNorm(cfg.d_model, **kw)
        if cfg.add_cross:
            self.cross = Attention(is_cross=True, **kw)
            self.norm_cross = qc.LayerNorm(cfg.d_model, **kw)
        d_ff = cfg.n_inner if cfg.n_inner is not None else 4 * cfg.d_model
        self.proj = openai.MLP(d_ff, **kw)
        self.norm = qc.LayerNorm(cfg.d_model, **kw)

    def forward(self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, **kw):
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, y_cache=True, yo=None)
        y = self.norm_attn(x)
        y, a, kv = self.attn(y, cache=cache, head_m=head_m, mask=mask, **kw)
        y = x + y
        a2 = None
        if enc is not None:
            x = y
            y = self.norm_cross(y)
            y, a2, kv2 = self.cross(y, enc_m=enc_m, enc=enc, head_m=head_m, mask=mask, **kw)
            y = x + y
            kv = kv + kv2
        x = y
        y = self.proj(self.norm(y))
        y = x + y
        y = (y,)
        if yo.attn:
            y += (a, a2)
        if yo.cache:
            y += (kv,)
        return y


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "drop_attn", "drop", "n_heads", "n_pos", "scale", "scale_by_inv"})

    def __init__(self, is_cross=False, lay_i=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        self.is_cross = is_cross
        self.lay_i = lay_i
        cfg = self.get_cfg(kw)
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        cfg.s_head = int(d / h)
        if is_cross:
            self.attn = qc.Conv1D(2 * d, d, **kw)
            self.query = qc.Conv1D(d, d, **kw)
        else:
            self.attn = qc.Conv1D(3 * d, d, **kw)
        self.proj = qc.Conv1D(d, d, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        p, t = cfg.n_pos, torch.bool
        self.register_buffer("bias", torch.tril(torch.ones((p, p), dtype=t)).view(1, 1, p, p))
        # self.register_buffer("bias_m", torch.tensor(-1e4))

    def forward(self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if enc is None:
            q, k, v = self.attn(x).split(cfg.d_model, dim=2)
        else:
            q = self.query(x)
            k, v = self.attn(enc).split(cfg.d_model, dim=2)
            mask = enc_m
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        if cfg.reorder:
            ys = self.reordered(q, k, v, mask, head_m, yo=yo)
        else:
            ys = self.scores(q, k, v, mask, head_m, yo=yo)
        y = self.join_heads(ys[0])
        y = (self.drop(self.proj(y)),)
        y += ys[1:]
        if yo.cache:
            y += ((k, v),)
        return y

    split_heads = qa.split_heads
    join_heads = qa.join_heads

    def scores(self, q, k, v, mask, head_m, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        a = torch.matmul(q, k.transpose(-1, -2))
        if cfg.scale:
            a = a / torch.full([], v.size(-1) ** 0.5, dtype=a.dtype, device=a.device)
        if cfg.scale_by_inv:
            a = a / float(self.lay_i + 1)
        if not self.is_cross:
            n_q, n_k = q.size(-2), k.size(-2)
            causal = self.bias[:, :, n_k - n_q : n_k, :n_k].bool()
            m = torch.tensor(torch.finfo(a.dtype).min, dtype=a.dtype).to(a.device)
            a = torch.where(causal, a, m)
        if mask is not None:
            a = a + mask
        a = self.drop_attn(F.softmax(a, dim=-1).type(v.dtype))
        if head_m is not None:
            a = a * head_m
        y = (torch.matmul(a, v),)
        if yo.attn:
            y += (a,)
        return y

    def reordered(self, q, k, v, mask, head_m, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        alpha = 1.0
        if cfg.scale:
            alpha /= float(v.size(-1)) ** 0.5
        if cfg.scale_by_inv:
            alpha /= float(self.lay_i + 1)
        b, h, n_q, d = q.size()
        _, _, n_k, _ = k.size()
        a = torch.empty(b * h, n_q, n_k, dtype=torch.float32, device=q.device)
        with autocast(enabled=False):
            q, k = q.reshape(-1, n_q, d), k.transpose(-1, -2).reshape(-1, d, n_k)
            a = torch.baddbmm(a, q.float(), k.float(), beta=0, alpha=alpha)
            a = a.reshape(b, h, n_q, n_k)
        if not self.is_cross:
            causal = self.bias[:, :, n_k - n_q : n_k, :n_k].bool()
            m = torch.tensor(torch.finfo(a.dtype).min, dtype=a.dtype).to(a.device)
            a = torch.where(causal, a, m)
        if mask is not None:
            a = a + mask
        a = self.drop_attn(F.softmax(a, dim=-1).type(v.dtype))
        if head_m is not None:
            a = a * head_m
        y = (torch.matmul(a, v),)
        if yo.attn:
            y += (a,)
        return y
