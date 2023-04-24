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

import numpy as np
import torch

from torch import nn
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core.embed import sin_embeds
from ..prep.config.ctrl import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw)
        self.pos_emb = qc.Embed(cfg.n_pos, m, **kw)
        sin_embeds(cfg.n_pos, m, out=self.pos_emb.weight)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.lays = qc.Stack(
            [Encoder(m, cfg.n_heads, cfg.d_ff, cfg.drop_resid) for _ in range(cfg.n_lays)]
        )
        self.norm = qc.LayerNorm(m, **kw)

    def forward(
        self, x, x_emb=None, prev_kv=None, mask=None, typ=None, pos=None, head_m=None, **kw
    ):
        cfg = self.cfg
        if x is not None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        b = s[0]
        d = x.device if x is not None else x_emb.device
        if prev_kv is None:
            past_length = 0
            prev_kv = tuple([None] * len(self.lays))
        else:
            past_length = prev_kv[0][0].size(-2)
        if pos is None:
            pos = torch.arange(past_length, s[-1] + past_length, dtype=torch.long, device=d)
            pos = pos.unsqueeze(0).view(-1, s[-1])
        if mask is not None:
            assert b > 0
            mask = mask.view(b, -1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.to(dtype=self.dtype)
            mask = (1.0 - mask) * -10000.0
        head_m = self.get_head_m(head_m, cfg.n_lays)
        if typ is None:
            typ = 0
        else:
            typ = self.tok_emb(typ.view(-1, s[-1])) * cfg.scale
        pos = pos.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x)
        # x_emb = embedded.unsqueeze(0) if len(x.shape)<2 else embedded
        n = s[-1]
        mask = torch.triu(torch.ones(n + past_length, n + past_length), 1).to(d)
        x_emb *= cfg.scale
        pos = self.pos_emb[pos, :].to(d)
        y = x_emb + pos + typ
        y = self.drop(y)
        attns = caches = hiddens = ()
        for i, (lay, layer_past) in enumerate(zip(self.lays, prev_kv)):
            hiddens += (y,)
            ys = lay(
                y,
                mask,
                layer_past=layer_past,
                mask=mask,
                head_m=head_m[i],
                **kw,
            )
            y, present = ys[:2]
            attns += (ys[2],)
            caches += (present,)
        y = self.norm(y)
        hiddens += (y,)
        return qo.WithCaches(y, attns, caches, hiddens)


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, bias=False, **kw)

    def forward(self, x, x_emb=None, labels=None, **kw):
        cfg = self.cfg
        ys = self.model(x, x_emb=x_emb, **kw)
        b = (x.shape[:2] if x is not None else x_emb.shape[:2])[0]
        if cfg.PAD is None:
            n = -1
        else:
            assert b == 1
            if x is not None:
                n = torch.ne(x, cfg.PAD).sum(-1) - 1
            else:
                n = -1
        y = self.proj(ys[0])[range(b), n]
        loss = None
        if labels is not None:
            if cfg.problem is None:
                dt = labels.dtype
                if cfg.n_labels == 1:
                    cfg.problem = "regression"
                elif cfg.n_labels > 1 and (dt == torch.long or dt == torch.int):
                    cfg.problem = "single_label"
                else:
                    cfg.problem = "multi_label"
            if cfg.problem == "regression":
                if cfg.n_labels == 1:
                    loss = nn.MSELoss()(y.squeeze(), labels.squeeze())
                else:
                    loss = nn.MSELoss()(y, labels)
            elif cfg.problem == "single_label":
                loss = nn.CrossEntropyLoss()(y.view(-1, cfg.n_labels), labels.view(-1))
            elif cfg.problem == "multi_label":
                loss = nn.BCEWithLogitsLoss()(y, labels)
        ys = (y,) + ys[2:] + (loss,)
        return qo.WithLoss(*ys)


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, bias=True, **kw)

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[..., :-1, :].contiguous()
            ls = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), ls.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCaches(*ys)


def scaled_dot_product_attention(q, k, v, mask, head_m=None):
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    dk = k.shape[-1]
    ys = matmul_qk / np.sqrt(dk)
    if mask is not None:
        nd, ns = ys.size(-2), ys.size(-1)
        ys += mask[ns - nd : ns, :ns] * -1e4
    if mask is not None:
        ys = ys + mask
    ys = torch.softmax(ys, dim=-1)
    if head_m is not None:
        ys = ys * head_m
    y = torch.matmul(ys, v)
    return y, ys


def point_wise_feed_forward_network(d_model, d_ff):
    return nn.Sequential(qc.Linear(d_model, d_ff), nn.ReLU(), qc.Linear(d_ff, d_model))


class Encoder(qc.Module):
    def __init__(self, d_model, n_heads, d_ff, rate=0.1):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.norm1 = qc.LayerNorm(d_model, eps=1e-6)
        self.norm2 = qc.LayerNorm(d_model, eps=1e-6)
        self.drop1 = qc.Dropout(rate)
        self.drop2 = qc.Dropout(rate)

    def forward(self, x, mask, layer_past=None, head_m=None, **kw):
        normed = self.norm1(x)
        ys = self.attn(
            normed, normed, normed, mask, layer_past=layer_past, mask=mask, head_m=head_m
        )
        out1 = x + self.drop1(ys[0])
        out2 = self.norm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.drop2(ffn_output)
        out2 = out1 + ffn_output
        y = (out2,) + ys[1:]
        return y


class Attention(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0  # or cfg.d_embed is not None
        cfg.d_head = h = m // n
        cfg.scale = 1 / (h**0.5)
        self.query = qc.Linear(m, m, **kw)
        self.key = qc.Linear(m, m, **kw)
        self.value = qc.Linear(m, m, **kw)
        self.proj = qc.Linear(m, m, **kw)

    def split_into_heads(self, x, batch_size):
        cfg = self.cfg
        x = x.reshape(batch_size, -1, cfg.n_heads, cfg.d_head)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask, layer_past=None, head_m=None, **kw):
        cfg = self.cfg
        b = q.shape[0]
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        q = self.split_into_heads(q, b)
        k = self.split_into_heads(k, b)
        v = self.split_into_heads(v, b)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = torch.stack((k, v))
        ys = scaled_dot_product_attention(q, k, v, mask, mask, head_m)
        scaled_attention = ys[0].permute([0, 2, 1, 3])
        attn = ys[1]
        original_size_attention = scaled_attention.reshape(b, -1, cfg.d_model)
        ys = self.proj(original_size_attention)
        return ys, present, attn
