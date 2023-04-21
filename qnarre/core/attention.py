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
from torch.nn.parameter import Parameter, UninitializedBuffer

from .. import core as qc
from . import utils as qu


def split_heads(self, x, k=False):
    cfg = self.cfg
    y = x.view(x.size()[:-1] + (cfg.n_heads, cfg.s_head))
    if k:
        return y.permute(0, 2, 3, 1)
    else:
        return y.permute(0, 2, 1, 3)


def join_heads(self, x):
    cfg = self.cfg
    y = x.permute(0, 2, 1, 3).contiguous()
    return y.view(y.size()[:-2] + (cfg.n_heads * cfg.s_head,))


class Attention(qc.Module):
    hs = qc.Hypers(
        ["d_model", "n_heads", "d_k", "d_v"],
        {
            "add_b_kv": False,
            "add_zero_attn": False,
            "batch_first": False,
            "bias": True,
            "drop": 0.0,
        },
    )

    w_pack, w_q, w_k, w_v = None
    b_pack, b_q, b_k, b_v = None

    def __init__(self, n_heads, d_model, hs=[], **kw):
        if n_heads is not None:
            kw.update(n_heads=n_heads)
        if d_model is not None:
            kw.update(d_model=d_model)
        super().__init__([self.hs] + hs, **kw)
        cfg = self.cfg
        n, h = cfg.n_heads, cfg.d_model
        assert h % n == 0
        d_k = cfg.d_k if cfg.d_k is not None else h
        d_v = cfg.d_v if cfg.d_v is not None else h
        self.pack = self.d_k == h and self.d_v == h
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        if self.pack:
            self.w_pack = Parameter(torch.empty((3 * h, h), **kw))
            self.register_parameter("w_q", None)
            self.register_parameter("w_k", None)
            self.register_parameter("w_v", None)
        else:
            self.register_parameter("w_pack", None)
            self.w_q = Parameter(torch.empty((h, h), **kw))
            self.w_k = Parameter(torch.empty((h, d_k), **kw))
            self.w_v = Parameter(torch.empty((h, d_v), **kw))
        if cfg.bias:
            self.b_pack = Parameter(torch.empty(3 * h, **kw))
        else:
            self.register_parameter("b_pack", None)
        self.out = Linear(h, h, bias=cfg.bias, **kw)
        if cfg.add_b_kv:
            self.b_k = Parameter(torch.empty((1, 1, h), **kw))
            self.b_v = Parameter(torch.empty((1, 1, h), **kw))
        else:
            self.register_parameter("b_k", None)
            self.register_parameter("b_v", None)

    def build(self, _):
        if not self.is_built():
            with torch.no_grad():
                self.reset_params()

    def reset_params(self):
        if self.pack:
            nn.init.xavier_uniform_(self.w_pack)
        else:
            nn.init.xavier_uniform_(self.w_q)
            nn.init.xavier_uniform_(self.w_k)
            nn.init.xavier_uniform_(self.w_v)
        if self.b_pack is not None:
            nn.init.constant_(self.b_pack, 0.0)
            nn.init.constant_(self.out.bias, 0.0)
        if self.b_k is not None:
            nn.init.xavier_normal_(self.b_k)
        if self.b_v is not None:
            nn.init.xavier_normal_(self.b_v)

    def forward(self, q, k, v, mask=None, k_mask=None, need_weights=True, average=True):
        cfg = self.cfg
        is_batched = q.dim() == 3
        if cfg.batch_first and is_batched:
            q, k, v = [x.transpose(1, 0) for x in (q, k, v)]
        if self.pack:
            y, w = self.multi_head_attention_forward(
                q,
                k,
                v,
                mask,
                k_mask,
                self.add_zero_attn,
                need_weights=need_weights,
                average=average,
            )
        else:
            y, w = self.multi_head_attention_forward(
                q,
                k,
                v,
                self.add_zero_attn,
                mask,
                k_mask,
                need_weights=need_weights,
                average=average,
            )
        if self.batch_first and is_batched:
            return y.transpose(1, 0), w
        else:
            return y, w

    def project_packed(self, q, k, v):
        w, b = self.w_pack, self.b_pack
        if k is v:
            if q is k:
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                H = q.size(-1)
                w_q, w_kv = w.split([H, H * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([H, H * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def project(self, q, k, v, bs):
        w_q, w_k, w_v = self.w_q, self.w_k, self.w_v
        H, Dk, Dv = q.size(-1), k.size(-1), v.size(-1)
        assert w_q.shape == (H, H) and w_k.shape == (H, Dk) and w_v.shape == (H, Dv)
        b_q, b_k, b_v = bs
        assert b_q is None or b_q.shape == (H,)
        assert b_k is None or b_k.shape == (H,)
        assert b_v is None or b_v.shape == (H,)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def attention(self, q, k, v, mask=None):
        cfg = self.cfg
        B, Nt, H = q.shape
        q = q / math.sqrt(H)
        w = torch.bmm(q, k.transpose(-2, -1))
        if mask is not None:
            w += mask
        w = softmax(w, dim=-1)
        if self.training and cfg.dropout_p > 0.0:
            w = drop(w, p=self.drop)
        y = torch.bmm(w, v)
        return y, w

    def is_batched(self, q, k, v, k_mask, mask):
        if q.dim() == 3:
            assert k.dim() == 3 and v.dim() == 3
            if k_mask is not None:
                assert k_mask.dim() == 2
            if mask is not None:
                assert mask.dim() in (2, 3)
            return True
        assert q.dim() == 2
        assert k.dim() == 2 and v.dim() == 2
        if k_mask is not None:
            assert k_mask.dim() == 1
        if mask is not None:
            assert mask.dim() in (2, 3)
            if mask.dim() == 3:
                assert mask.shape == (self.cfg.n_heads, q.shape[0], k.shape[0])
        return False

    def multi_head_attention_forward(
        self,
        q,
        k,
        v,
        mask=None,
        k_mask=None,
        add_zero_attn=None,
        need_weights=True,
        static_k=None,
        static_v=None,
        average=True,
    ):
        if not self.is_batched(q, k, v, k_mask, mask):
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            if k_mask is not None:
                k_mask = k_mask.unsqueeze(0)
        cfg = self.cfg
        h, n = cfg.d_model, cfg.n_heads
        b_q, b_k, b_v = self.b_q, self.b_k, self.b_v
        if self.pack:
            assert k.shape == v.shape
            q, k, v = self.project_packed(q, k, v)
        else:
            assert k.shape[:2] == v.shape[:2]
            if self.b_pack is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.b_pack.chunk(3)
            q, k, v = self.project(q, k, v, (b_q, b_k, b_v))
        d_tgt, d_batch, _ = q.shape
        d_src, _, _ = k.shape
        if mask is not None:
            assert mask.is_floating_point() or mask.dtype == torch.bool
            if mask.dim() == 2:
                assert mask.shape == (d_tgt, d_src)
                mask = mask.unsqueeze(0)
            else:
                assert mask.shape == (d_batch * n, d_tgt, d_src)
        if b_k is not None and b_v is not None:
            assert static_k is None
            assert static_v is None
            k = torch.cat([k, b_k.repeat(1, d_batch, 1)])
            v = torch.cat([v, b_v.repeat(1, d_batch, 1)])
            if mask is not None:
                mask = pad(mask, (0, 1))
            if k_mask is not None:
                k_mask = pad(k_mask, (0, 1))
        else:
            assert b_k is None
            assert b_v is None
        d_head = h // n
        q = q.contiguous().view(d_tgt, d_batch * n, d_head).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], d_batch * n, d_head).transpose(0, 1)
        else:
            assert static_k.size(0) == d_batch * n
            assert static_k.size(2) == d_head
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], d_batch * n, d_head).transpose(0, 1)
        else:
            assert static_v.size(0) == d_batch * n
            assert static_v.size(2) == d_head
            v = static_v
        if add_zero_attn:
            zero_attn_shape = (d_batch * n, 1, d_head)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if mask is not None:
                mask = pad(mask, (0, 1))
            if k_mask is not None:
                k_mask = pad(k_mask, (0, 1))
        d_src = k.size(1)
        if k_mask is not None:
            assert k_mask.shape == (d_batch, d_src)
            k_mask = (
                k_mask.view(d_batch, 1, 1, d_src)
                .expand(-1, n, -1, -1)
                .reshape(d_batch * n, 1, d_src)
            )
            if mask is None:
                mask = k_mask
            elif mask.dtype == torch.bool:
                mask = mask.logical_or(k_mask)
            else:
                mask = mask.masked_fill(k_mask, float("-inf"))
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(mask, float("-inf"))
        y, w = _scaled_dot_product_attention(q, k, v, mask)
        y = y.transpose(0, 1).contiguous().view(d_tgt, d_batch, h)
        y = F.linear(y, self.out.weight, self.out.bias)
        if need_weights:
            w = w.view(d_batch, n, d_tgt, d_src)
            if average:
                w = w.sum(dim=1) / n
            return y, w
        else:
            return y, None


class Attend(qc.Module):
    hs = qc.Hypers(
        [
            "d_attn_k",
            "d_attn_v",
            "d_attn",
            "d_model",
            "drop_attn",
            "drop",
            "len_mem",
            "n_heads",
            "pos_type",
            "proxim_bias",
        ],
        {},
    )
    v_w = pos_tim = proxim_b = None

    def __init__(self, owner, hs=[], **kw):
        super().__init__([self.hs] + hs, **kw)
        self.owner = owner
        self.pre = owner.pre
        self.post = owner.post
        self.pos_x_b = owner.pos_x_b
        self.pos_p_b = owner.pos_p_b
        cfg = self.cfg
        h, n = cfg.d_model, cfg.n_heads
        assert h % n == 0
        k = cfg.d_attn_k or cfg.d_attn or h
        assert k % n == 0
        self.scale = 1 / (k**0.5)
        v = cfg.d_attn_v or k
        assert v % n == 0
        kw = {"dtype": cfg.dtype, "device": cfg.device}
        if k == v:
            self.qkv_w = Parameter(torch.empty((h, n * k), **kw))
        else:
            self.qk_w = Parameter(torch.empty((h, n * k), **kw))
            self.v_w = Parameter(torch.empty((h, n * v), **kw))
        self.out_w = Parameter(torch.empty((n * v, h), **kw))
        if cfg.pos_type == "relative":
            self.pos_tim = PosTiming(**kw)
            self.pos_w = Parameter(torch.empty((h, n * k), **kw))
            if self.pos_x_b is None:
                self.pos_x_b = Parameter(torch.empty((n, k), **kw))
            if self.pos_p_b is None:
                self.pos_p_b = Parameter(torch.empty((n, k), **kw))
        if cfg.proxim_bias:
            self.proxim_b = Proximity(**kw)

    def build(self, x):
        if not self.is_built():
            cfg = self.cfg
            with torch.no_grad():
                e = x.shape[1] + cfg.len_mem if cfg.len_mem else 0
                if cfg.pos_type == "relative":
                    self.pos_tim.materialize(cfg.d_model, e)
                if cfg.proxim_bias:
                    self.proxim_b.materialize(e)

    def reset_params(self):
        if self.is_built():
            a = math.sqrt(5)
            if self.v_w is None:
                nn.init.kaiming_uniform_(self.qkv_w, a=a)
            else:
                nn.init.kaiming_uniform_(self.qk_w, a=a)
                nn.init.kaiming_uniform_(self.v_w, a=a)
            if self.pos_tim is not None:
                nn.init.kaiming_uniform_(self.pos_w, a=a)
            if self.owner.pos_x_b is None:
                nn.init.kaiming_uniform_(self.pos_x_b, a=a)
            if self.owner.pos_p_b is None:
                nn.init.kaiming_uniform_(self.pos_p_b, a=a)

    split_heads = split_heads

    join_heads = join_heads

    def forward(self, x, mask=None):
        x, ctx = x[0], x[1] if len(x) > 1 else None
        xlen = x.shape[1]
        y = x if ctx is None else torch.cat([ctx, x], dim=1)
        y = self.pre([y, y])
        if self.v_w is None:
            y = v = torch.einsum("bih,hk->bik", y, self.qkv_w)
        else:
            y = torch.einsum("bih,hk->bik", y, self.qk_w)
            v = torch.einsum("bih,hv->biv", v, self.v_w)
        q = self.split_heads(y[:, -xlen:, :])
        k = self.split_heads(y)
        if self.pos_tim is None:
            qk = torch.einsum("bnik,bnjk->bnij", q, k)
        else:
            qk = self.to_qk_with_pos(q, k)
        v = self.split_heads(v)
        y = self.to_scores(qk, mask, v)
        y = self.join_heads(y)
        y = torch.einsum("biv,vh->bih", y, self.out_w)
        y = self.post([x, y])
        return y

    def to_qk_with_pos(self, q, k):
        b = self.pos_x_b[:, None, :, None]
        y = torch.einsum("bnik,bnjk->bnij", q + b, k)
        p = torch.einsum("ih,hk->ik", self.pos_tim, self.pos_w)
        # fmt: off
        p = self.split_heads(p)[None,]
        # fmt: on
        b = self.pos_p_b[:, None, :, None]
        p = torch.einsum("bnik,bnjk->bnij", q + b, p)
        y += self.shift(p)
        return y

    def shift(self, x):
        s = x.shape
        y = torch.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        y = torch.reshape(y, [s[0], s[1], s[3] + 1, s[2]])
        y = torch.slice(y, [0, 0, 1, 0], [-1, -1, -1, -1])
        y = torch.reshape(y, s)
        return y

    def to_scores(self, qk, mask, v):
        b = 0
        if mask is not None:
            b = torch.logical_not(mask)
            b = torch.cast(b, torch.floatx()) * qu.big_neg()
            if self.proxim_b is not None:
                b += self.proxim_b
            b = b[:, None, :, None]
        y = torch.softmax(qk * self.scale + b)
        cfg = self.cfg
        y = self.drop(y, cfg.drop_attn or cfg.drop)
        y = torch.einsum("bnij,bnjv->bniv", y, v)
        return y


class Proximity(UninitializedBuffer):
    def materialize(self, end, dtype=None, device=None):
        dtype = self.data.dtype if dtype is None else dtype
        device = self.data.device if device is None else device
        kw = {"dtype": dtype, "device": device}
        y = torch.arange(end, **kw)
        # fmt: off
        y = (y[None,] - y[:, None])
        y = -torch.log1p(torch.abs(y))
        self.data = y[None, None,]
        # fmt: on
        self.__class__ = self.cls_to_become


class PosTiming(UninitializedBuffer):
    def materialize(self, dim, end, dtype=None, device=None):
        dtype = self.data.dtype if dtype is None else dtype
        device = self.data.device if device is None else device
        kw = {"dtype": dtype, "device": device}
        t = torch.arange(end - 1, -1, -1.0, **kw)
        f = torch.arange(0, dim, 2.0, **kw)
        f = 1 / (10000 ** (f / dim))
        t = torch.einsum("i,d->id", t, f)
        self.data = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        self.__class__ = self.cls_to_become


class PosTiming(UninitializedBuffer):
    def materialize(self, dim, end, p_max, p_min, p_start, dtype=None, device=None):
        dtype = self.data.dtype if dtype is None else dtype
        device = self.data.device if device is None else device
        kw = {"dtype": dtype, "device": device}
        t = torch.arange(end, **kw) + p_start
        assert dim % 2 == 0
        n = dim // 2
        f = np.log(p_max / p_min) / max(n - 1, 1)
        f = torch.arange(n, **kw) * -f
        f = torch.exp(f) * p_min
        t = torch.einsum("i,d->id", t, f)
        self.data = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        self.__class__ = self.cls_to_become
