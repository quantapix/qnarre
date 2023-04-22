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

import torch

from torch import nn

from .. import core as qc
from . import utils as qu


class GPT(qc.Module):
    hs = qc.Hypers({"act", "d_ff", "d_model", "drop"}, {})

    def __init__(self, d_ff=None, ps={}, hs=[], **kw):
        if d_ff is not None:
            kw.update(d_ff=d_ff)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.ff = qc.Conv1D(cfg.d_ff, cfg.d_model, **kw)
        self.proj = qc.Conv1D(cfg.d_model, cfg.d_ff, **kw)
        self.act = qu.activation(cfg.act)
        self.drop = nn.Dropout(cfg.drop, **kw)

    def forward(self, x):
        y = self.ff(x)
        y = self.act(y)
        y = self.proj(y)
        y = self.drop(y)
        return y


class FFNet(qc.Module):
    hs = qc.Hypers({"act", "d_ff", "d_model", "drop", "eps", "chunk_ff"}, {"seq_len_dim": 1})

    def __init__(self, act=None, drop=None, eps=None, ps={}, hs=[], **kw):
        if act is not None:
            kw.update(act=act)
        if drop is not None:
            kw.update(drop=drop)
        if eps is not None:
            kw.update(eps=eps)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.ff = qc.Linear(cfg.d_model, cfg.d_ff, **kw)
        self.act = None if cfg.act is None else qu.activation(cfg.act)
        self.drop = None if cfg.drop is None else qc.Dropout(cfg.drop, **kw)
        self.proj = qc.Linear(cfg.d_ff, cfg.d_model, **kw)
        self.norm = None if cfg.eps is None else qc.LayerNorm(cfg.d_model, cfg.eps, **kw)

    def forward(self, *xs):
        cfg = self.cfg
        size, dim = cfg.chunk_ff, cfg.seq_len_dim
        assert len(xs) > 0
        if size > 0:
            shape = xs[0].shape[dim]
            for x in xs:
                assert x.shape[dim] == shape
            assert xs[0].shape[dim] % size == 0
            n = xs[0].shape[dim] // size
            ys = tuple(x.chunk(n, dim=dim) for x in xs)
            ys = tuple(self.chunker(*y) for y in zip(*ys))
            return torch.cat(ys, dim=dim)
        return self.chunker(*xs)

    def chunker(self, x):
        y = self.ff(x)
        if self.act:
            y = self.act(y)
        # if self.drop:
        #    y = self.drop(y)
        y = self.proj(y)
        if self.drop:
            y = self.drop(y)
        if self.norm:
            y = self.norm(y + x)
        return y


class Masker(qc.Module):
    hs = qc.Hypers({"d_model", "d_lin", "eps", "s_vocab"}, {"act": "gelu"})

    def __init__(self, d_lin=None, act=None, ps={}, hs=[], **kw):
        if d_lin is not None:
            kw.update(d_lin=d_lin)
        if act is not None:
            kw.update(act=act)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        n = cfg.d_lin or m
        self.lin = qc.Linear(m, n, **kw)
        self.act = qu.activation(cfg.act)
        self.norm = qc.LayerNorm(n, cfg.eps, **kw)
        self.proj = qc.Linear(n, cfg.s_vocab, bias=False, **kw)
        self.bias = nn.Parameter(torch.zeros(cfg.s_vocab))
        self.proj.bias = self.bias

    def forward(self, x):
        y = self.lin(x)
        y = self.act(y)
        y = self.norm(y)
        y = self.proj(y)
        return y


class Classifier(qc.Module):
    hs = qc.Hypers({"d_model", "d_lin", "drop", "drop_proj", "n_labels"}, {"act": "tanh"})

    def __init__(self, d_lin=None, act=None, **kw):
        if d_lin is not None:
            kw.update(d_lin=d_lin)
        if act is not None:
            kw.update(act=act)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        if cfg.d_lin is None:
            self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)
        else:
            n = cfg.d_lin
            self.lin = qc.Linear(cfg.d_model, n, **kw)
            self.act = qu.activation(cfg.act)
            self.proj = qc.Linear(n, cfg.n_labels, **kw)
        p = cfg.drop_proj if cfg.drop_proj is not None else cfg.drop
        self.drop = qc.Dropout(p, **kw)

    def forward(self, x):
        y = x  # [:, 0, :] take <s> token (equiv. to [CLS])
        if self.cfg.d_lin is not None:
            y = self.drop(y)
            y = self.lin(y)
            y = self.act(y)
        y = self.drop(y)
        y = self.proj(y)
        return y


class Pool(qc.Module):
    hs = qc.Hypers(["d_model"], {})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lin = qc.Linear(cfg.d_model, cfg.d_model, **kw)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.lin(x[:, 0])
        y = self.act(y)
        return y


class PoolBeg(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = qc.Linear(cfg.d_model, 1)

    def forward(self, x, mask=None):
        y = self.proj(x).squeeze(-1)
        if mask is not None:
            if self.get_param_dtype() == torch.float16:
                y = y * (1 - mask) - 65500 * mask
            else:
                y = y * (1 - mask) - 1e30 * mask
        return y


class PoolEnd(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff = qc.Linear(cfg.d_model * 2, cfg.d_model)
        self.act = nn.Tanh()
        self.norm = qc.LayerNorm(cfg.d_model, cfg.eps)
        self.proj = qc.Linear(cfg.d_model, 1)

    def forward(self, x, x_beg=None, beg_pos=None, mask=None):
        assert x_beg is not None or beg_pos is not None
        if beg_pos is not None:
            slen, hsz = x.shape[-2:]
            beg_pos = beg_pos[:, None, None].expand(-1, -1, hsz)
            x_beg = x.gather(-2, beg_pos)
            x_beg = x_beg.expand(-1, slen, -1)
        y = self.ff(torch.cat([x, x_beg], dim=-1))
        y = self.act(y)
        y = self.norm(y)
        y = self.proj(y).squeeze(-1)
        if mask is not None:
            if self.get_param_dtype() == torch.float16:
                y = y * (1 - mask) - 65500 * mask
            else:
                y = y * (1 - mask) - 1e30 * mask
        return y


class PoolProj(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff = qc.Linear(cfg.d_model * 2, cfg.d_model)
        self.act = nn.Tanh()
        self.proj = qc.Linear(cfg.d_model, 1, bias=False)

    def forward(self, x, x_beg=None, beg_pos=None, idx=None):
        hsz = x.shape[-1]
        assert x_beg is not None or beg_pos is not None
        if beg_pos is not None:
            beg_pos = beg_pos[:, None, None].expand(-1, -1, hsz)
            x_beg = x.gather(-2, beg_pos).squeeze(-2)
        if idx is not None:
            idx = idx[:, None, None].expand(-1, -1, hsz)
            cls_token_state = x.gather(-2, idx).squeeze(-2)
        else:
            cls_token_state = x[:, -1, :]
        y = self.ff(torch.cat([x_beg, cls_token_state], dim=-1))
        y = self.act(y)
        y = self.proj(y).squeeze(-1)
        return y


class Positionwise(qc.Module):
    hs = qc.Hypers({"d_ff", "d_model", "drop"}, {"eps": 1e-5, "pre_norm": False})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, f = cfg.d_model, cfg.d_ff
        self.ff = nn.Sequential(
            qc.Linear(m, f, **kw),
            nn.ReLU(inplace=True),
            qc.Dropout(cfg.drop, **kw),
            qc.Linear(f, m, **kw),
            qc.Dropout(cfg.drop, **kw),
        )
        self.norm = qc.LayerNorm(m, **kw)

    def forward(self, x):
        if self.cfg.pre_norm:
            return x + self.ff(self.norm(x))
        return self.norm(x + self.ff(x))
