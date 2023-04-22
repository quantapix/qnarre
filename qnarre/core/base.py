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

import itertools
import math
import numbers
import operator
import torch

from collections import OrderedDict, abc
from dataclasses import dataclass
from itertools import chain
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import is_lazy, Parameter, UninitializedParameter

from . import utils as qu


class Hypers:
    @staticmethod
    def merge(hs):
        y = Hypers()
        for h in hs:
            y.ks.update(h.ks)
            y.kw.update(h.kw)
        return y

    def __init__(self, ks=None, kw=None):
        self.ks = ks or set()
        self.kw = kw or dict()

    def __repr__(self):
        return f"Hypers(**{self.__dict__})"


class Config:
    @staticmethod
    def merge(ps={}, h=Hypers(), **kw):
        ps = ps.__dict__ if isinstance(ps, Config) else ps
        assert isinstance(ps, dict)
        for k in h.ks:
            yield k, ps.get(k),
        for k, v in h.kw.items():
            yield k, ps.get(k, v)
        for k, v in kw.items():
            yield k, v

    def __init__(self, ps={}, h=Hypers(), **kw):
        super().__init__()
        for k, v in Config.merge(ps, h, **kw):
            setattr(self, k, v)

    def __repr__(self):
        return f"Config(**{self.__dict__})"


@dataclass
class Yopts:
    attn = None
    cache = None
    hidden = None
    kw = None


class Module(nn.Module):
    hs = Hypers({"dtype", "device"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__()
        self.cfg = Config(ps, Hypers.merge([self.hs] + hs), **kw)

    def get_cfg(self, kw=None):
        y = self.cfg
        if kw is not None:
            kw.update(ps=y)
        return y

    def get_y_opts(self, y_attn=None, y_cache=None, y_hidden=None, y_kw=None, yo=None, **_):
        cfg = self.cfg
        if yo is not None:
            y_attn = y_attn if yo.attn is None else yo.attn
            y_cache = y_cache if yo.cache is None else yo.cache
            y_hidden = y_hidden if yo.hidden is None else yo.hidden
            y_kw = y_kw if yo.kw is None else yo.kw
        y_attn = cfg.y_attn if y_attn is None else y_attn
        y_cache = cfg.y_cache if y_cache is None else y_cache
        y_hidden = cfg.y_hidden if y_hidden is None else y_hidden
        y_kw = cfg.y_kw if y_kw is None else y_kw
        return Yopts(y_attn, y_cache, y_hidden, y_kw)

    def invert_mask(self, x):
        if x.dim() == 3:
            y = x[:, None, :, :]
        if x.dim() == 2:
            y = x[:, None, None, :]
        dt = self.cfg.dtype
        y = y.to(dtype=dt)
        if dt == torch.float16:
            y = (1.0 - y) * -1e4
        else:
            assert dt in [torch.bfloat16, torch.float32]
            y = (1.0 - y) * -1e9
        return y

    def get_param_dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:

            def fn(x):
                return [(k, v) for k, v in x.__dict__.items() if torch.is_tensor(v)]

            y = next(self._named_members(get_members_fn=fn))
            return y[1].dtype

    def get_minus_inf(self):
        return -65000 if self.get_param_dtype() == torch.float16 else -1e30


class Identity(Module):
    def __init__(self, *xs, **kw):
        super().__init__(*xs, **kw)

    def forward(self, x, **_):
        return x


class Linear(Module):
    hs = Hypers({"d_in", "d_out"}, {"bias": True})

    def __init__(self, d_in=None, d_out=None, ps={}, hs=[], **kw):
        if d_in is not None:
            kw.update(d_in=d_in)
        if d_out is not None:
            kw.update(d_out=d_out)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        kw = {"dtype": cfg.dtype, "device": cfg.device}
        self.weight = Parameter(torch.empty((cfg.d_out, cfg.d_in), **kw))
        if cfg.bias:
            self.bias = Parameter(torch.empty(cfg.d_out, **kw))
        else:
            self.register_parameter("bias", None)
        self.reset_params()

    def reset_params(self):
        cfg = self.cfg
        b = 1 / math.sqrt(cfg.d_in)
        # nn.init.uniform_(self.weight, -b, b)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -b, b)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Bilinear(Module):
    hs = Hypers({"d_in1", "d_in2", "d_out"}, {"bias": True})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        kw = {"dtype": cfg.dtype, "device": cfg.device}
        self.weight = Parameter(torch.empty((cfg.d_out, cfg.d_in1, cfg.d_in2), **kw))
        if cfg.bias:
            self.bias = Parameter(torch.empty(cfg.d_out, **kw))
        else:
            self.register_parameter("bias", None)
        self.reset_params()

    def reset_params(self):
        cfg = self.cfg
        b = 1 / math.sqrt(cfg.d_in1)
        nn.init.uniform_(self.weight, -b, b)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -b, b)

    def forward(self, x1, x2):
        return F.linear(x1, x2, self.weight, self.bias)


class Embed(Module):
    hs = Hypers(
        {"d_embed", "max_norm", "n_embed", "PAD"},
        {"norm_type": 2.0, "scale_grad": False, "sparse": False},
    )

    def __init__(self, n_embed=None, d_embed=None, ps={}, hs=[], **kw):
        if n_embed is not None:
            kw.update(n_embed=n_embed)
        if d_embed is not None:
            kw.update(d_embed=d_embed)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        if cfg.PAD is not None:
            if cfg.PAD > 0:
                assert cfg.PAD < cfg.n_embed
            elif cfg.PAD < 0:
                assert cfg.PAD >= -cfg.n_embed
                cfg.PAD = self.n_embed + cfg.PAD
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        self.weight = Parameter(torch.empty((cfg.n_embed, cfg.d_embed), **kw))
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.weight)
        cfg = self.cfg
        if cfg.PAD is not None:
            with torch.no_grad():
                self.weight[cfg.PAD].fill_(0)

    def forward(self, x):
        c = self.cfg
        return F.embedding(x, self.weight, c.PAD, c.max_norm, c.norm_type, c.scale_grad, c.sparse)

    @classmethod
    def from_data(cls, x, freeze=True, **kw):
        y = cls(**kw)
        y.weight.data = x
        y.weight.requires_grad = not freeze
        return y


class LayerNorm(Module):
    hs = Hypers(kw={"eps": 1e-5, "elemwise_affine": True})

    def __init__(self, shape, eps=None, ps={}, hs=[], **kw):
        if eps is not None:
            kw.update(eps=eps)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        self.shape = tuple(shape)
        kw = {"device": cfg.device, "dtype": cfg.dtype}
        if cfg.elemwise_affine:
            self.weight = Parameter(torch.empty(self.shape, **kw))
            self.bias = Parameter(torch.empty(self.shape, **kw))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_params()

    def reset_params(self):
        if self.cfg.elemwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.layer_norm(x, self.shape, self.weight, self.bias, self.cfg.eps)


class Dropout(Module):
    hs = Hypers(kw={"p": 0.5, "inplace": False})

    def __init__(self, p=None, ps={}, hs=[], **kw):
        if p is not None:
            kw.update(p=p)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        assert cfg.p >= 0 and cfg.p <= 1

    def forward(self, x):
        cfg = self.cfg
        return F.drop(x, cfg.p, self.training, cfg.inplace)


class Stack(Module):
    _modules = OrderedDict()

    def __init__(self, xs=None, **kw):
        super().__init__(**kw)
        if xs is not None:
            self += xs

    def _get_abs_string_index(self, i):
        i = operator.index(i)
        if not (-len(self) <= i < len(self)):
            raise IndexError("index {} is out of range".format(i))
        if i < 0:
            i += len(self)
        return str(i)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(list(self._modules.values())[i])
        else:
            return self._modules[self._get_abs_string_index(i)]

    def __setitem__(self, i, x):
        i = self._get_abs_string_index(i)
        return setattr(self, str(i), x)

    def __delitem__(self, i):
        if isinstance(i, slice):
            for k in range(len(self._modules))[i]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(i))
        x = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(x, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, xs):
        return self.extend(xs)

    def __add__(self, x):
        y = Stack()
        for i, module in enumerate(chain(self, x)):
            y.add_module(str(i), module)
        return y

    def __dir__(self):
        y = super(Stack, self).__dir__()
        y = [k for k in y if not k.isdigit()]
        return y

    def insert(self, i, x):
        for j in range(len(self._modules), i, -1):
            self._modules[str(j)] = self._modules[str(j - 1)]
        self._modules[str(i)] = x

    def append(self, x):
        self.add_module(str(len(self)), x)
        return self

    def extend(self, xs):
        if not isinstance(xs, abc.Iterable):
            raise TypeError("extend needs to be called with an iterable")
        off = len(self)
        for i, x in enumerate(xs):
            self.add_module(str(off + i), x)
        return self


class Lazy(Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._init_hook = self.register_forward_pre_hook(self._builder)
        self._load_hook = self._register_load_state_dict_pre_hook(self._loader)

    def _builder(self, m, x):
        m.build(*x)
        if not self.is_built():
            raise RuntimeError("Not fully built")
        m._load_hook.remove()
        m._init_hook.remove()
        delattr(m, "_init_hook")
        delattr(m, "_load_hook")

    def is_built(self):
        for v in itertools.chain(self._parameters.values(), self._buffers.values()):
            if is_lazy(v):
                return False
        return True

    def build(self):
        raise NotImplementedError()

    def _loader(self, state, pre):
        for k, v in itertools.chain(self._parameters.items(), self._buffers.items()):
            k = pre + k
            if k in state and v is not None:
                x = state[k]
                if is_lazy(v):
                    if not is_lazy(x):
                        with torch.no_grad():
                            v.materialize(x.shape)

    def _save_to_state_dict(self, dst, pre, keep):
        for n, p in self._parameters.items():
            if p is not None:
                if not (is_lazy(p) or keep):
                    p = p.detach()
                dst[pre + n] = p
        for n, b in self._buffers.items():
            if b is not None and n not in self._non_persistent_buffers_set:
                if not (is_lazy(b) or keep):
                    b = b.detach()
                dst[pre + n] = b

    def _replicate_for_data_parallel():
        raise RuntimeError("Not fully built")


class LazyLin(Lazy):
    hs = Hypers({"d_in", "d_out"}, {"bias": True})

    def __init__(self, d_out=None, ps={}, hs=[], **kw):
        if d_out is not None:
            kw.update(d_out=d_out)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        kw = {"dtype": cfg.dtype, "device": cfg.device}
        self.weight = UninitializedParameter(**kw)
        if cfg.bias:
            self.bias = UninitializedParameter(**kw)
        else:
            self.register_parameter("bias", None)

    def build(self, x):
        cfg = self.cfg
        if not self.is_built():
            with torch.no_grad():
                cfg.d_in = x.shape[-1]
                self.weight.materialize((cfg.d_out, cfg.d_in))
                if cfg.bias:
                    self.bias.materialize((cfg.d_out,))
                self.reset_params()

    def reset_params(self):
        cfg = self.cfg
        if self.is_built():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            b = 1 / math.sqrt(cfg.d_in)
            # nn.init.uniform_(self.weight, -b, b)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -b, b)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Conv1D(Module):
    hs = Hypers()

    def __init__(self, n_y, n_x, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        cfg.n_y = n_y
        self.weight = Parameter(torch.empty(n_x, n_y))
        self.bias = Parameter(torch.zeros(n_y))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        s = x.size()[:-1] + (self.cfg.n_y,)
        y = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        y = y.view(s)
        return y


class SeqSummary(Module):
    hs = Hypers()

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.cfg
        self.summy_type = getattr(cfg, "summy_type", "last")
        if self.summy_type == "attn":
            raise NotImplementedError
        self.summy = Identity()
        if hasattr(cfg, "sum_use_proj") and cfg.sum_use_proj:
            if hasattr(cfg, "sum_proj") and cfg.sum_proj and cfg.num_labels > 0:
                num_classes = cfg.num_labels
            else:
                num_classes = cfg.d_model
            self.summy = nn.Linear(cfg.d_model, num_classes)
        activation_string = getattr(cfg, "sum_act", None)
        self.act = qu.activation(activation_string, Identity())
        self.drop_1 = Identity()
        if hasattr(cfg, "drop_sum_first") and cfg.drop_sum_first > 0:
            self.drop_1 = nn.Dropout(cfg.drop_sum_first)
        self.drop_2 = Identity()
        if hasattr(cfg, "summary_last_dropout") and cfg.summary_last_dropout > 0:
            self.drop_2 = nn.Dropout(cfg.summary_last_dropout)

    def forward(self, x, idx=None):
        if self.summy_type == "last":
            y = x[:, -1]
        elif self.summy_type == "first":
            y = x[:, 0]
        elif self.summy_type == "mean":
            y = x.mean(dim=1)
        elif self.summy_type == "cls_index":
            if idx is None:
                idx = torch.full_like(
                    x[..., :1, :],
                    x.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                idx = idx.unsqueeze(-1).unsqueeze(-1)
                idx = idx.expand((-1,) * (idx.dim() - 1) + (x.size(-1),))
            y = x.gather(-2, idx).squeeze(-2)
        elif self.summy_type == "attn":
            raise NotImplementedError
        y = self.drop_1(y)
        y = self.summy(y)
        y = self.act(y)
        y = self.drop_2(y)
        return y


if __name__ == "__main__":
    h0 = Hypers()
    print(h0)
    h1 = Hypers({"a", "b"}, {"c": 3, "d": 4})
    print(h1)
    c0 = Config()
    print(c0)
    c1 = Config({"b": 20})
    print(c1)
    c2 = Config({"b": 20}, h1, d=40)
    print(c2)
    c3 = Config({"b": 20}, Hypers.merge([h1, Hypers(set(), {"a": 100})]), c=300, d=40)
    print(c3)
    c4 = Config(c3, Hypers.merge([h1, Hypers({"bbb"}, {"a": 0, "aaa": 0})]), c=0, ddd=0)
    print(c4)
