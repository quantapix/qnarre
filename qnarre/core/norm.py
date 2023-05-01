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

from qnarre.core.base import Module

from .. import core as qc
from . import utils as qu


class RMS(qc.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            y = y.to(self.weight.dtype)
        return self.weight * y


def _layer_norm(self, inputs):
    x = inputs
    m = torch.reduce_mean(x, axis=-1, keepdims=True)
    v = torch.reduce_mean(torch.square(x - m), axis=-1, keepdims=True)
    y = (x - m) / torch.sqrt(v + self.cfg.eps)
    y = y * self.norm_w + self.norm_b
    return y


class Norm(qc.Module):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                "eps",
            )
        )

    def build(self, input_shape):
        s = input_shape[-1]
        self.norm_w = self.add_weight("norm_w", s, initializer="ones")
        self.norm_b = self.add_weight("norm_b", s, initializer="zeros")
        return super().build(input_shape)

    def forward(self, inputs):
        return _layer_norm(self, inputs)


class LayerProc(Module):
    cmd = ""
    batch = None

    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                "bdims_prepost",
                "cmd_post",
                "cmd_pre",
                "drop",
                "drop_prepost",
                "eps",
                "norm_type",
            )
        )

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        if cfg.norm_type == "batch":
            self.batch = torch.BatchNormalization(epsilon=cfg.eps)

    def build(self, input_shape):
        s = input_shape[1][-1]
        self.norm_w = self.add_weight("norm_w", s, initializer="ones")
        self.norm_b = self.add_weight("norm_b", s, initializer="zeros")
        # self.gamma = self.add_weight(shape=(), initializer='zeros')
        return super().build(input_shape)

    def forward(self, inputs):
        prev, x = inputs
        y = x
        if self.cmd:
            cfg = self.cfg
            for c in self.cmd:
                if c == "a":
                    y = prev + x
                elif c == "z":
                    y = prev + x * self.gamma
                elif c == "n":
                    if cfg.norm_type == "layer":
                        y = _layer_norm(self, x)
                    elif cfg.norm_type == "batch":
                        y = self.batch(x)
                    elif cfg.norm_type == "l2":
                        m = torch.reduce_mean(x, axis=-1, keepdims=True)
                        n = torch.square(x - m)
                        n = torch.reduce_sum(n, axis=-1, keepdims=True)
                        y = (x - m) / torch.sqrt(n + cfg.eps)
                        y = y * self.gain + self.bias
                    elif cfg.norm_type == "group":
                        sh = torch.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % cfg.n_groups == 0
                        gs = (cfg.n_groups, sh[-1] // cfg.n_groups)
                        x = torch.reshape(x, sh[:-1] + gs)
                        m, v = torch.moments(x, [1, 2, 4], keep_dims=True)
                        y = (x - m) / torch.sqrt(v + cfg.group_eps)
                        y = torch.reshape(y, sh) * self.gain + self.bias
                    elif cfg.norm_type == "noam":
                        y = torch.cast_to_floatx(torch.int_shape(x)[-1])
                        y = torch.l2_normalize(x, axis=-1) * torch.sqrt(y)
                    else:
                        assert cfg.norm_type == "none"
                else:
                    assert c == "d"
                    y = self.drop(y)
                x = y
        return y

    def drop(self, x):
        cfg = self.cfg
        r = cfg.drop_prepost or cfg.drop
        ns, ds = None, [int(i) for i in cfg.bdims_prepost.split(",") if i]
        if ds:
            sh = ()
            n = len(sh)
            ds = [d + n if d < 0 else d for d in ds]
            ns = [1 if i in ds else sh[i] for i in range(n)]
        return super().drop(x, r, noise_shape=ns)


class PreProc(LayerProc):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        self.cmd = self.cfg.cmd_pre
        assert "a" not in self.cmd
        assert "z" not in self.cmd


class PostProc(LayerProc):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        self.cmd = self.cfg.cmd_post


import torch
from torch.nn import LayerNorm as LayerNorm


def get_norm(neox_args):
    if neox_args.norm == "rmsnorm":
        norm = RMSNorm
        eps = neox_args.rms_norm_epsilon
    elif neox_args.norm == "layernorm":
        eps = neox_args.layernorm_epsilon
        norm = LayerNorm
    elif neox_args.norm == "scalenorm":
        eps = neox_args.scalenorm_epsilon
        norm = ScaleNorm
    else:
        raise ValueError(f"norm {neox_args.norm} not recognized")
    return norm, eps


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g
