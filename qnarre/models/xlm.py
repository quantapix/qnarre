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
# https://arxiv.org/abs/1911.02116
# https://arxiv.org/abs/2105.00572

import itertools
import torch

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.activations import gelu
from transformers.utils import logging


from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.embed import sin_embeds
from ..core.mlp import Classifier, MLP

from ..prep.config.xlm import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        assert not self.is_dec
        cfg.d_embed = cfg.d_emb  # 512 by default
        cfg.d_model = cfg.d_embed * 4  # 2048 by default
        assert cfg.d_embed % cfg.n_heads == 0
        self.pos_emb = qc.Embed(cfg.n_pos, cfg.d_embed, **kw)
        if cfg.sin_embeds:
            sin_embeds(cfg.n_pos, cfg.d_embed, out=self.pos_emb.weight)
        if cfg.n_langs > 1 and cfg.use_lang_emb:
            self.lang_emb = qc.Embed(self.n_langs, cfg.d_embed, **kw)
        self.tok_emb = qc.Embed(self.s_vocab, cfg.d_embed, **kw)
        self.norm_emb = qc.LayerNorm(cfg.d_embed, cfg.eps, **kw)
        self.attns = qc.Stack()
        self.norm1 = qc.Stack()
        self.ffnet = qc.Stack()
        self.norm2 = qc.Stack()
        for _ in range(self.n_lays):
            self.attns.append(Attention(cfg.n_heads, cfg.d_embed, **kw))
            self.norm1.append(qc.LayerNorm(cfg.d_embed, cfg.eps, **kw))
            self.ffnet.append(
                MLP(
                    gelu if cfg.gelu_activation else F.relu,
                    drop=cfg.drop,
                    d_model=cfg.d_embed,
                    d_ff=cfg.d_model,
                    **kw,
                )
            )
            self.norm2.append(qc.LayerNorm(cfg.d_embed, cfg.eps, **kw))
        self.register_buffer("pos", torch.arange(cfg.n_pos).expand((1, -1)))

    def forward(
        self,
        x,
        cache=None,
        head_m=None,
        langs=None,
        lengths=[],
        mask=None,
        pos=None,
        typ=None,
        x_emb=None,
        **kw,
    ):
        cfg = self.cfg
        if x is None:
            b, n = x_emb.size()[:-1]
        else:
            b, n = x.size()
        d = x.device if x is not None else x_emb.device
        if lengths is None:
            if x is None:
                lengths = torch.tensor([n] * b, device=d)
            else:
                lengths = (x != cfg.PAD).sum(dim=1).long()
        assert lengths.size(0) == b
        assert lengths.max().item() <= n
        mask, attn_mask = get_masks(n, lengths, cfg.causal, mask=mask)
        if pos is None:
            pos = self.pos[:, :n]
        else:
            assert pos.size() == (b, n)
        if langs is not None:
            assert langs.size() == (b, n)
        head_m = self.get_head_m(head_m, cfg.n_lays)
        if cache is not None and x is not None:
            _slen = n - cache["slen"]
            x = x[:, -_slen:]
            pos = pos[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        if x_emb is None:
            x_emb = self.tok_emb(x)
        y = x_emb + self.pos_emb(pos).expand_as(x_emb)
        if langs is not None and self.use_lang_emb and cfg.n_langs > 1:
            y = y + self.lang_emb(langs)
        if typ is not None:
            y = y + self.tok_emb(typ)
        y = self.norm_emb(y)
        y = F.drop(y, p=self.drop, training=self.training)
        y *= mask.unsqueeze(-1).to(y.dtype)
        attns = hiddens = ()
        for i in range(cfg.n_lays):
            hiddens += (y,)
            ys = self.attns[i](y, attn_mask, cache=cache, head_m=head_m[i], **kw)
            y = ys[0]
            attns += (ys[1],)
            y = F.drop(y, p=cfg.drop, training=self.training)
            y = y + y
            y = self.norm1[i](y)
            y = y + self.ffnet[i](y)
            y = self.norm2[i](y)
            y *= mask.unsqueeze(-1).to(y.dtype)
        hiddens += (y,)
        if cache is not None:
            cache["slen"] += y.size(1)
        return qo.Base(y, attns, hiddens)


class ForMulti(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.sum = qc.SeqSummary(**kw)
        self.proj = qc.Linear(cfg.n_labels, 1, **kw)

    def forward(self, x, mask=None, langs=None, typ=None, pos=None, x_emb=None, labels=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask, typ, pos, langs = qu.view_2D(x, mask, typ, pos, langs)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x=x, mask=mask, langs=langs, typ=typ, pos=pos, x_emb=x_emb, **kw)
        y = self.proj(self.sum(ys[0])).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForQASimple(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.SQuADHead(**kw)

    def forward(
        self, x, beg_pos=None, end_pos=None, is_impossible=None, cls_index=None, p_mask=None, **kw
    ):
        ys = self.model(x, **kw)
        y = self.proj(
            ys[0],
            beg_pos=beg_pos,
            end_pos=end_pos,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
            **kw,
        )
        ys = (y[0],) + ys[1:] + y[1:]
        return QATop(*ys)


@dataclass
class QATop(qc.Output):
    logits: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    loss: tuple = None
    top_beg = None
    top_beg_i = None
    top_end = None
    top_end_i = None


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.SeqSummary(**kw)

    forward = qf.forward_seq


class ForTokClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Projector(**kw)

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0], labels)
        y = y[0] if labels is None else y[1]
        loss = y[0] if labels is not None else None
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class Projector(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.asm = cfg.asm
        if cfg.asm is False:
            self.proj = qc.Linear(cfg.d_embed, cfg.s_vocab, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=cfg.d_embed,
                n_classes=cfg.s_vocab,
                cutoffs=cfg.asm_cutoffs,
                div_value=cfg.asm_div_value,
                head_bias=True,
            )

    def forward(self, x, y=None):
        if self.asm is False:
            ys = (self.proj(x),)
            if y is not None:
                loss = F.cross_entropy(ys.view(-1, self.s_vocab), y.view(-1), reduction="mean")
                ys = (loss,) + ys
        else:
            ys = (self.proj.log_prob(x),)
            if y is not None:
                _, loss = self.proj(x, y)
                ys = (loss,) + ys
        return ys


def get_masks(slen, lengths, causal, mask=None):
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if mask is None:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]
    b = lengths.size(0)
    if causal:
        attn_mask = alen[None, None, :].repeat(b, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask
    assert mask.size() == (b, slen)
    assert causal is False or attn_mask.size() == (b, slen, slen)
    return mask, attn_mask


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads"}, {"drop_attn": 0.0})
    NEW_ID = itertools.count()

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        self.lay_id = next(Attention.NEW_ID)
        cfg = self.get_cfg(kw)
        n, d = cfg.n_heads, cfg.d_model
        cfg.s_head = int(d / n)
        cfg.scale = cfg.s_head**-0.5
        assert d % n == 0
        self.query = qc.Linear(d, d, **kw)
        self.key = qc.Linear(d, d, **kw)
        self.value = qc.Linear(d, d, **kw)
        self.drop = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(d, d, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        q = self.split_heads(self.query(x) * cfg.scale)
        if enc is None:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
            if cache is not None:
                if self.lay_id in cache:
                    k = torch.cat([cache[self.lay_id][0], k], dim=2)
                    v = torch.cat([cache[self.lay_id][1], v], dim=2)
        else:
            if cache is None or self.lay_id not in cache:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
            else:
                k, v = cache[self.lay_id]
        # ??? cache[self.lay_id] = (k, v)
        y = torch.matmul(q, k.transpose(2, 3))
        b, qlen, _ = x.size()
        if enc is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = enc.size(1)
        s = (b, 1, qlen, klen) if mask.dim() == 3 else (b, 1, 1, klen)
        mask = (mask == 0).view(s).expand_as(y)
        y.masked_fill_(mask, -float("inf"))
        y = self.drop(F.softmax(y.float(), dim=-1).type_as(y))
        if head_m is not None:
            y = y * head_m
        a = y
        y = torch.matmul(y, v)
        y = qa.join_heads(y)
        return self.proj(y), a, cache
