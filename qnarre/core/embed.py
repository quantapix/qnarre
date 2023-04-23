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

import math
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .. import core as qc
from . import utils as qu


class Embeds(qc.Module):
    hs = qc.Hypers(
        {"d_embed", "drop", "eps", "n_pos", "n_typ", "s_vocab"},
        {"pos_type": "absolute", "rescale": False},
    )

    def __init__(self, d_embed=None, ps={}, hs=[], **kw):
        if d_embed is not None:
            kw.update(d_embed=d_embed)
        super().__init__([self.hs] + hs, ps, **kw)
        cfg = self.get_cfg(kw)
        self.tok = qc.Embed(cfg.s_vocab, cfg.d_embed, **kw)
        if cfg.n_pos is not None:
            self.pos = qc.Embed(cfg.n_pos, cfg.d_embed, **kw)
            self.register_buffer("pos_ids", torch.arange(cfg.n_pos).expand((1, -1)))
            if cfg.pos_sin:
                sin_embeds(n_pos=cfg.n_pos, dim=cfg.d_embed, out=self.pos.weight)
        if cfg.n_typ is not None:
            s = self.pos_ids.size()
            self.typ = qc.Embed(cfg.n_typ, cfg.d_embed, **kw)
            self.register_buffer("typ_ids", torch.zeros(s, dtype=torch.long), persistent=False)
        self.norm = qc.LayerNorm(cfg.d_embed, cfg.eps, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, y=None, typ=None, pos=None, kv_len=0, **_):
        cfg = self.cfg
        if y is None:
            y = self.tok(x)
            s = x.size()
        else:
            s = y.size()[:-1]
        b, n = s[:2]
        if cfg.rescale:
            y = y * (cfg.d_embed**0.5)
        if typ is None:
            if hasattr(self, "typ_ids"):
                typ = self.typ_ids[:, :n].expand(b, n)
            else:
                typ = torch.zeros(s, dtype=torch.long, device=self.pos_ids.device)
        y = y + self.typ(typ)
        if self.pos_type == "absolute":
            if pos is None:
                # roberta
                if x is None:
                    pos = self.create_pos(y)
                else:
                    p = self.cfg.PAD
                    mask = x.ne(p).int()
                    pos = (torch.cumsum(mask, dim=1).type_as(mask) + kv_len) * mask
                    pos = pos.long() + p
                # reoberta end

                if hasattr(self, "pos_ids"):
                    pos = self.pos_ids[:, kv_len : n + kv_len]
                else:
                    pos = (
                        torch.arange(n, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
                    )
            y = y + self.pos(pos)
        y = self.norm(y)
        y = self.drop(y)
        return y

    def create_pos(self, x):
        s = x.size()[:-1]
        n = s[1]
        p = self.cfg.PAD
        y = torch.arange(p + 1, n + p + 1, dtype=torch.long, device=x.device)
        return y.unsqueeze(0).expand(s)


def sin_embeds(n_pos, dim, y):
    y.requires_grad = False
    pos = np.array(
        [[i / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for i in range(n_pos)]
    )
    y[:, 0::2] = torch.FloatTensor(np.sin(pos[:, 0::2]))
    y[:, 1::2] = torch.FloatTensor(np.cos(pos[:, 1::2]))
    y.detach_()


def pos_enc(cfg, qlen, klen, bsz=None):
    freq_seq = torch.arange(0, cfg.d_model, 2.0, dtype=torch.float)
    inv_freq = 1 / torch.pow(10000, (freq_seq / cfg.d_model))
    if cfg.attn_type == "bi":
        beg, end = klen, -qlen
    elif cfg.attn_type == "uni":
        beg, end = klen, -1
    else:
        raise ValueError(f"Unknown `attn_type` {cfg.attn_type}.")

    def pos_emb(pos_seq, inv_freq, batch=None):
        seq = torch.einsum("i,d->id", pos_seq, inv_freq)
        y = torch.cat([torch.sin(seq), torch.cos(seq)], dim=-1)
        y = y[:, None, :]
        if batch is not None:
            y = y.expand(-1, batch, -1)
        return y

    if cfg.bi_data:
        fwd = torch.arange(beg, end, -1.0, dtype=torch.float)
        bwd = torch.arange(-beg, -end, 1.0, dtype=torch.float)
        if cfg.clamp_len > 0:
            fwd = fwd.clamp(-cfg.clamp_len, cfg.clamp_len)
            bwd = bwd.clamp(-cfg.clamp_len, cfg.clamp_len)
        if bsz is not None:
            fwd_pos = pos_emb(fwd, inv_freq, bsz // 2)
            bwd_pos = pos_emb(bwd, inv_freq, bsz // 2)
        else:
            fwd_pos = pos_emb(fwd, inv_freq)
            bwd_pos = pos_emb(bwd, inv_freq)
        y = torch.cat([fwd_pos, bwd_pos], dim=1)
    else:
        fwd = torch.arange(beg, end, -1.0)
        if cfg.clamp_len > 0:
            fwd = fwd.clamp(-cfg.clamp_len, cfg.clamp_len)
        y = pos_emb(fwd, inv_freq, bsz)
    y = y.to(cfg.device)
    return y


class TokEmbed(qc.Module):
    hs = qc.Hypers(
        {"brackets", "d_embed", "d_model", "one_hot", "max_norm", "s_vocab", "PAD"},
        {"norm_type": 2.0, "scale_grad": False, "sparse": False},
    )

    def __init__(self, s_vocab=None, d_embed=None, hs=[], **kw):
        if s_vocab is not None:
            kw.update(s_vocab=s_vocab)
        if d_embed is not None:
            kw.update(d_embed=d_embed)
        kw.update(hs=[self.hs] + hs)
        super().__init__(**kw)
        cfg = self.cfg
        h = cfg.d_model
        d = cfg.d_embed or h
        bs = (cfg.brackets or []) + [cfg.s_vocab]
        b = 0
        self.weights = []
        self.adjusts = []
        assert b == cfg.PAD
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        for i, e in enumerate(bs):
            p = d // (len(bs) ** i)
            t = Parameter(torch.empty((e - b, p), **kw))
            self.weights.append(t)
            a = None if p == h else Parameter(torch.empty((p, h), **kw))
            self.adjusts.append(a)
            b = e
        self.one_hot = cfg.one_hot

    def build(self):
        self.reset_params()

    def reset_params(self):
        for w in self.weights:
            nn.init.normal_(w)
        for a in self.adjusts:
            if a is not None:
                nn.init.normal_(a)
        cfg = self.cfg
        if cfg.PAD is not None:
            with torch.no_grad():
                for w in self.weights:
                    w[cfg.PAD].fill_(0)

    def forward(self, x):
        cfg = self.cfg
        y = torch.zeros(torch.int_shape(x) + (cfg.d_model,))
        bs = (cfg.brackets or []) + [cfg.s_vocab]
        b = 0
        for i, e in enumerate(bs):
            m = (x >= (b or 1)) & (x < e)
            u = torch.boolean_mask(x, m)
            u = self.lookup(u - b, i)
            y = torch.tensor_scatter_nd_add(y, torch.where(m), u)
            b = e
        y *= y.shape[-1] ** 0.5
        y.mask = torch.not_equal(x, cfg.PAD)
        return y

    def lookup(self, x, i):
        t = self.weights[i]
        if self.one_hot:
            y = torch.one_hot(x, torch.shape(t)[0], axis=-1)
            y = torch.einsum("np,in->ip", t, y)
        else:
            cfg = self.cfg
            y = F.embedding(x, t, cfg.PAD, cfg.max_norm, cfg.norm_type, cfg.scale_grad, cfg.sparse)
        a = self.adjusts[i]
        if a is not None:
            y = torch.einsum("ip,ph->ih", y, a)
        return y


class PosEmbed(qc.Embed):
    def __init__(self, n_embed, d_embed):
        self.offset = 2
        super().__init__(n_embed + self.offset, d_embed)

    def forward(self, shape, kv_len=0):
        _, n = shape[:2]
        x = torch.arange(kv_len, kv_len + n, dtype=torch.long, device=self.weight.device)
        return super().forward(x + self.offset)


class PosEmbed2(qc.Module):
    hs = qc.Hypers({"d_model", "d_src", "d_tgt", "pos_max_len"})

    def __init__(self, n_typ=None, d_embed=None, hs=[], **kw):
        if n_typ is not None:
            kw.update(n_typ=n_typ)
        if d_embed is not None:
            kw.update(d_embed=d_embed)
        kw.update(hs=[self.hs] + hs)
        super().__init__(**kw)
        cfg = self.cfg
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        self.pos_b = Parameter(torch.empty((cfg.n_typ, cfg.d_model), **kw))

    def build(self):
        cfg = self.cfg
        p = max(cfg.pos_max_len or 0, cfg.d_src, cfg.d_tgt)
        self.pos_b = self.add_weight("pos_b", (p, cfg.d_model))

    def forward(self, x, mask=None):
        y = self.pos_b[: x.shape[1], :]
        if mask is not None:
            y *= torch.cast(mask, self.pos_b.dtype)
        return x + y


class PosTiming(qc.Module):
    hs = qc.Hypers(
        {"d_model", "pos_max", "pos_min", "pos_start", "d_src", "d_tgt", "pos_max_len"},
    )

    def build(self):
        cfg = self.cfg
        m = cfg.d_model
        p = max(cfg.pos_max_len or 0, cfg.d_src, cfg.d_tgt)
        a = (cfg.pos_max, cfg.pos_min, cfg.pos_start)
        self.pos_b = qu.pos_timing(m, p, *a)

    def forward(self, x, mask=None):
        y = self.pos_b
        if mask is not None:
            y *= torch.cast(mask, self.pos_b.dtype)
        return x + y


class RelEmbed(qc.Module):
    hs = qc.Hypers({"d_model", "d_src", "d_tgt", "pos_max_len"})

    def __init__(self, n_typ=None, d_embed=None, hs=[], **kw):
        if n_typ is not None:
            kw.update(n_typ=n_typ)
        if d_embed is not None:
            kw.update(d_embed=d_embed)
        kw.update(hs=[self.hs] + hs)
        super().__init__(**kw)
        cfg = self.cfg
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        self.pos_b = Parameter(torch.empty((cfg.n_typ, cfg.d_model), **kw))

    def build(self):
        cfg = self.cfg
        p = max(cfg.pos_max_len or 0, cfg.d_src, cfg.d_tgt)
        self.pos_b = qu.PosTiming(cfg.d_model, p)

    def forward(self, x, mask=None):
        y = self.pos_b
        if mask is not None:
            y *= torch.cast(mask, self.pos_b.dtype)
        return x + y


class Positional(qc.Module):
    hs = qc.Hypers({"d_embed"})

    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_embed
        self.register_buffer("inv_freq", 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d)))

    def forward(self, x, b=None):
        y = torch.ger(x, self.inv_freq)
        y = torch.cat([y.sin(), y.cos()], dim=-1)[:, None, :]
        return y if b is None else y.expand(-1, b, -1)


class Adaptive(qc.Module):
    def __init__(self, s_vocab, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super().__init__()
        cfg = self.cfg
        cfg.scale = d_proj**0.5
        cfg.cutoffs = cutoffs + [s_vocab]
        cfg.ends = [0] + cfg.cutoffs
        self.lays = qc.Stack()
        self.projs = nn.ParameterList()
        if div_val == 1:
            self.lays.append(qc.Embed(s_vocab, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            for i in range(len(cfg.cutoffs)):
                left, right = cfg.ends[i], cfg.ends[i + 1]
                d = d_embed // (div_val**i)
                self.lays.append(qc.Embed(right - left, d))
                self.projs.append(nn.Parameter(torch.FloatTensor(d_proj, d)))

    def forward(self, x):
        cfg = self.cfg
        if cfg.div_val == 1:
            y = self.lays[0](x)
            if cfg.d_proj != cfg.d_embed:
                y = F.linear(y, self.projs[0])
        else:
            p = next(self.parameters())
            y = x.view(-1)
            ys = torch.zeros([y.size(0), cfg.d_proj], dtype=p.dtype, device=p.device)
            for i in range(len(cfg.cutoffs)):
                left, right = cfg.ends[i], cfg.ends[i + 1]
                mask = (y >= left) & (y < right)
                j = mask.nonzero().squeeze()
                if j.numel() == 0:
                    continue
                k = self.lays[i](y.index_select(0, j) - left)
                k = F.linear(k, self.projs[i])
                ys.index_copy_(0, j, k)
            y = ys.view(x.size() + (cfg.d_proj,))
        y.mul_(self.scale)
        return y


class SinEmbed(qc.Embed):
    def __init__(self, n_pos, d_embed, PAD):
        self.make_weight(n_pos, d_embed, PAD)

    def make_weight(self, n_pos, d_embed, PAD):
        w = self.get_embedding(n_pos, d_embed, PAD)
        if not hasattr(self, "weight"):
            super().__init__(n_pos, d_embed, PAD, _weight=w)
        else:
            w = w.to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight = nn.Parameter(w)
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(n_embed, d_embed, PAD):
        half = d_embed // 2
        y = math.log(10000) / (half - 1)
        y = torch.exp(torch.arange(half, dtype=torch.float) * -y)
        y = torch.arange(n_embed, dtype=torch.float).unsqueeze(1) * y.unsqueeze(0)
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=1).view(n_embed, -1)
        if d_embed % 2 == 1:
            y = torch.cat([y, torch.zeros(n_embed, 1)], dim=1)
        if PAD is not None:
            y[PAD, :] = 0
        return y

    @staticmethod
    def make_positions(x, PAD):
        mask = x.ne(PAD).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + PAD

    def forward(self, x, incremental_state=None, timestep=None):
        b, n = x.shape[:2]
        max_pos = self.PAD + 1 + n
        if max_pos > self.weight.size(0):
            self.make_weight(max_pos, cfg.d_embed, self.PAD)
        pos = self.make_positions(x, self.PAD)
        return super().forward(pos)


class SinEmbed2(qc.Embed):
    def __init__(self, n_pos, d_embed, PAD=None):
        super().__init__(n_pos, d_embed)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(y):
        n, d = y.shape
        pos = np.array(
            [[i / np.power(10000, 2 * (j // 2) / d) for j in range(d)] for i in range(n)]
        )
        y.requires_grad = False
        sentinel = d // 2 if d % 2 == 0 else (d // 2) + 1
        y[:, 0:sentinel] = torch.FloatTensor(np.sin(pos[:, 0::2]))
        y[:, sentinel:] = torch.FloatTensor(np.cos(pos[:, 1::2]))
        y.detach_()
        return y

    @torch.no_grad()
    def forward(self, shape, kv_len=0):
        b, n = shape[:2]
        pos = torch.arange(kv_len, kv_len + n, dtype=torch.long, device=self.weight.device)
        return super().forward(pos)


class RotaryEmbed(qc.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    y1 = x[..., : x.shape[-1] // 2]
    y2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-y2, y1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, xs):
    ys = xs[:, None, :, None]  # [bs, 1, seq_len, 1]
    ys = ys.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(ys.shape[0], 1, 1, 1), 2, ys)
    sin = torch.gather(sin.repeat(ys.shape[0], 1, 1, 1), 2, ys)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
