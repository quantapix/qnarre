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


class Embed(qc.Module):
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
            self.register_buffer("pos", torch.arange(cfg.n_pos).expand((1, -1)))
            if cfg.pos_sin:
                sin_embed(n_pos=cfg.n_pos, dim=cfg.d_embed, out=self.pos.weight)
        if cfg.n_typ is not None:
            s = self.pos.size()
            self.typ = qc.Embed(cfg.n_typ, cfg.d_embed, **kw)
            self.register_buffer("typ", torch.zeros(s, dtype=torch.long), persistent=False)
        self.norm = qc.LayerNorm(cfg.d_embed, cfg.eps, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, n_kv=0, pos=None, typ=None, x_emb=None, **_):
        cfg = self.cfg
        if x_emb is None:
            x_emb = self.tok(x)
            s = x.size()
        else:
            s = x_emb.size()[:-1]
        if cfg.rescale:
            x_emb = x_emb * (cfg.d_embed**0.5)
        b, n = s[:2]
        if typ is None:
            if hasattr(self, "typ"):
                typ = self.typ[:, :n].expand(b, n)
            else:
                typ = torch.zeros(s, dtype=torch.long, device=self.pos.device)
        y = x_emb + self.typ(typ)
        if cfg.pos_type == "absolute":
            if pos is None:
                p = self.cfg.PAD
                if hasattr(self, "pos"):
                    pos = self.pos[:, n_kv : n + n_kv]
                elif x is None:
                    pos = torch.arange(p + 1, n + p + 1, dtype=torch.long, device=x.device)
                    pos = pos.unsqueeze(0).expand_as(x)
                else:
                    mask = x.ne(p).int()
                    pos = (torch.cumsum(mask, dim=1).type_as(mask) + n_kv) * mask
                    pos = pos.long() + p
            y = y + self.pos(pos)
        y = self.norm(self.drop(y))
        return y


def sin_embed(n_pos, dim, y):
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

    def forward(self, x, n_kv=0):
        b, n = x.shape[:2]
        y = torch.arange(n_kv, n_kv + n, dtype=torch.long, device=self.weight.device).expand(b, -1)
        return super().forward(y + self.offset)


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
    def forward(self, shape, n_kv=0):
        b, n = shape[:2]
        pos = torch.arange(n_kv, n_kv + n, dtype=torch.long, device=self.weight.device)
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


# Copyright (c) 2021, EleutherAI
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

import torch
import math


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def bias(self, seq_len_q, seq_len_k, device, dtype):
        # [b, np, sq, sk]
        # seq_len_q = x.shape[-2]
        # seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            a = -torch.tril(
                torch.arange(target_seq_len).view(target_seq_len, 1).repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(device).to(dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return a

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            a = -torch.tril(
                torch.arange(target_seq_len).view(target_seq_len, 1).repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a


# Copyright (c) 2021, EleutherAI
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

import torch
import math
from torch.nn.parameter import Parameter

from megatron import mpu
from megatron.model.positional_embeddings import SinusoidalPositionalEmbedding
from megatron.model.init_functions import get_init_methods


class Embedding(torch.nn.Module):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        neox_args,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_pos_emb=True,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.use_mup = neox_args.use_mup
        self.mup_embedding_mult = neox_args.mup_embedding_mult
        self.mup_rp_embedding_mult = neox_args.mup_rp_embedding_mult

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            neox_args=neox_args,
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
            init_method=self.init_method,
        )
        self._word_embeddings_key = "word_embeddings"

        if neox_args.use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                self.embedding_module = bnb.nn.StableEmbedding
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            self.embedding_module = torch.nn.Embedding

        # Position embedding (serial).
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.embedding_type = neox_args.pos_emb
            if self.embedding_type == "learned":
                self.position_embeddings = self.embedding_module(
                    max_sequence_length, self.hidden_size
                )
                self._position_embeddings_key = "position_embeddings"
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.weight)
            elif self.embedding_type == "sinusoidal":
                self.position_embeddings = SinusoidalPositionalEmbedding(self.hidden_size)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = self.embedding_module(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.opt_pos_emb_offset = neox_args.opt_pos_emb_offset

        # For ticking position ids forward
        self.layer_past = None

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print("adding embedding for {} tokentypes".format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = self.embedding_module(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.use_pos_emb and self.embedding_type in ["learned", "sinusoidal"]:
            if self.opt_pos_emb_offset:
                if self.layer_past is not None:
                    position_ids = position_ids + self.layer_past + 1
                self.layer_past = position_ids[:, -1]
                # OPT always adds 2 for some reason, according to the HF implementation
                position_ids = position_ids + self.opt_pos_emb_offset
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings.mul_(self.mup_rp_embedding_mult)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        if self.use_mup:
            with torch.no_grad():
                embeddings.mul_(self.mup_embedding_mult)

        return embeddings


class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    def forward(self, args):
        assert (
            len(args) == 3
        ), f"Expected 3 arguments (input_ids, position_ids, attention_mask), but got {len(args)}."

        input_ids = args[0]
        position_ids = args[1]
        attention_mask = args[2]
        embeddings = super().forward(input_ids, position_ids)
        return embeddings, attention_mask


class SoftEmbedding(torch.nn.Module):
    def __init__(
        self,
        neox_args,
        wte,
        n_tokens: int = 10,
        init_range: float = 0.5,
        init_string: str = "",
    ):
        super(SoftEmbedding, self).__init__()
        self.n_tokens = n_tokens
        self.neox_args = neox_args
        self.init_range = init_range
        self.init_string = init_string
        self.soft_embedding_weight = torch.nn.parameter.Parameter(self.initialize_embedding(wte))

    def initialize_embedding(self):
        if self.init_string:
            embeds = torch.LongTensor(self.neox_args.tokenizer.tokenize(self.init_string)).to(
                self.embedding_module.weight.device
            )
            embeds = self.embedding_module(embeds)
            if embeds.shape[0] >= self.n_tokens:
                embeds = embeds[: self.n_tokens, :]  # slice
            else:
                embeds = embeds.repeat(math.ceil(self.n_tokens / embeds.shape[0]), 1)[
                    : self.n_tokens, :
                ]  # pad up to n_tokens
            return embeds
        return torch.Tensor(n_tokens, neox_args.hidden_size).uniform_(
            -self.random_range, self.random_range
        )

    def forward(self, args: tuple):
        in_inference = len(args) == 3  # embeddings, layer_past, attention_mask
        in_train = len(args) == 2  # embeddings, attention_mask
        if in_train:
            embedding, attention_mask = args
        else:
            embedding, layer_past, attention_mask = args
        soft_embedding = self.soft_embedding_weight.repeat(
            embedding.shape[0], 1, 1
        )  # repeat batch_size times
        if in_train:
            # append soft embedding at the beginning in training
            embedding = torch.cat((soft_embedding, embedding), dim=1)
            embedding = embedding[:, : self.neox_args.seq_length, ...]
            return embedding, attention_mask
        else:
            if not (exists(layer_past) and layer_past.numel() > 0):
                # if in inference, on the first forward pass, we want to do the same as in training (append soft embedding)
                embedding = torch.cat((soft_embedding, embedding), dim=1)
                embedding = embedding[:, : self.neox_args.seq_length, ...]
            # otherwise, we're in incremental mode, and just want to forward the single embedding (since the soft prompt has already been cached)
            return embedding, layer_past, attention_mask
