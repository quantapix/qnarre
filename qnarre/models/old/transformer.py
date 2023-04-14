import copy
import torch

from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

from .base import Hypers, Module, Lazy, Stack, Linear, LayerNorm, Dropout
from .attention import Attention


class Transformer(Lazy):
    hs = Hypers(
        [],
        {
            "activation": F.relu,
            "batch_first": False,
            "d_ffnet": 2048,
            "d_hidden": 512,
            "depth_dec": 6,
            "depth_enc": 6,
            "drop": 0.1,
            "n_heads": 8,
            "norm_eps": 1e-5,
            "norm_first": False,
        },
    )

    def __init__(self, depth=None, hs=[], **kw):
        if depth is not None:
            kw.update(depth_enc=depth, depth_dec=depth)
        super().__init__([self.hs] + hs, **kw)
        cfg = self.cfg
        kw.update(ps=cfg)
        h, e, d = cfg.d_hidden, cfg.depth_enc, cfg.depth_dec
        n = LayerNorm(h, cfg.norm_eps, **kw)
        self.e_stack = EncStack(self, e, Encoder(**kw), n, **kw)
        n = LayerNorm(h, cfg.norm_eps, **kw)
        self.d_stack = DecStack(self, d, Decoder(**kw), n, **kw)

    def build(self, x):
        cfg = self.cfg
        if not self.is_built():
            with torch.no_grad():
                self.reset_params()

    def forward(self, x, mask, k_mask):
        cfg = self.cfg
        batched = src.dim() == 3
        if not cfg.batch_first and src.size(1) != tgt.size(1) and batched:
            raise RuntimeError("batches mismatch")
        elif cfg.batch_first and src.size(0) != tgt.size(0) and batched:
            raise RuntimeError("batches mismatch")
        if src.size(-1) != cfg.d_hidden or tgt.size(-1) != cfg.d_hidden:
            raise RuntimeError("d_hidden mismatch")
        src, tgt = x
        ctx = self.e_stack([src], src_mask, src_k_mask)
        y = self.d_stack([tgt, ctx], [tgt_mask, ctx_mask], [tgt_k_mask, ctx_k_mask])
        return y

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Stack(Module):
    hs = Hypers([], {"depth": 2})

    def __init__(self, owner, depth=None, hs=[], **kw):
        if depth is not None:
            kw.update(depth=depth)
        super().__init__([self.hs] + hs, **kw)
        self.pre = owner.pre
        self.post = owner.post


class EncStack(Stack):
    def __init__(self, owner, depth, enc, norm=None, **kw):
        super().__init__(owner, depth, **kw)
        self.encs = _clones(enc, self.cfg.depth)
        self.norm = norm

    def forward(self, x, mask=None, k_mask=None):
        x = x[0]
        y = self.pre([x, x])
        for e in self.encs:
            y = e([y], mask, k_mask)
        if self.norm is not None:
            y = self.norm(y)
        y = self.post([x, y])
        return y


class DecStack(Stack):
    def __init__(self, owner, depth, dec, norm=None, **kw):
        super().__init__(owner, depth, **kw)
        self.decs = _clones(dec, self.cfg.depth)
        self.norm = norm

    def forward(self, x, mask=None, k_mask=None):
        x, ctx = x
        y = self.pre([x, x])
        for d in self.decs:
            y = d([y, ctx], mask, k_mask)
        if self.norm is not None:
            y = self.norm(y)
        y = self.post([x, y])
        return y


class Encoder(Module):
    hs = Hypers(
        [],
        {
            "activation": F.relu,
            "batch_first": False,
            "d_ffnet": 2048,
            "d_hidden": 512,
            "drop": 0.1,
            "norm_eps": 1e-5,
            "n_heads": 2,
            "norm_first": False,
        },
    )

    def __init__(self, hs=[], **kw):
        super().__init__([self.hs] + hs, **kw)
        cfg = self.cfg
        kw.update(ps=cfg)
        n, h = cfg.n_heads, cfg.d_hidden

        self.refl = Attention(n, h, **kw)

        self.lin1 = Linear(cfg.d_ffnet, **kw)
        self.active = cfg.activation
        self.drop = Dropout()
        self.lin2 = Linear(h, **kw)

        self.norm1 = LayerNorm(h, cfg.norm_eps, **kw)
        self.norm2 = LayerNorm(h, cfg.norm_eps, **kw)
        self.drop1 = Dropout()
        self.drop2 = Dropout()

    def __setstate__(self, x):
        if "activation" not in x:
            x["activation"] = F.relu
        super(Encoder, self).__setstate__(x)

    def forward(self, src, mask=None, k_mask=None):
        x = src
        if self.cfg.norm_first:
            x = x + self.reflect(self.norm1(x), mask, k_mask)
            x = x + self.ffnet(self.norm2(x))
        else:
            x = self.norm1(x + self.reflect(x, mask, k_mask))
            x = self.norm2(x + self.ffnet(x))
        return x

    def reflect(self, x, mask, k_mask):
        x = self.refl(x, x, x, mask, k_mask, need_weights=False)[0]
        return self.drop1(x)

    def ffnet(self, x):
        x = self.lin2(self.drop(self.activ(self.lin1(x))))
        return self.drop2(x)


class Decoder(Module):
    hs = Hypers(
        [],
        {
            "activation": F.relu,
            "batch_first": False,
            "d_ffnet": 2048,
            "d_hidden": 512,
            "drop": 0.1,
            "norm_eps": 1e-5,
            "n_heads": 2,
            "norm_first": False,
        },
    )

    def __init__(self, hs=[], **kw):
        super().__init__([self.hs] + hs, **kw)
        cfg = self.cfg
        kw.update(ps=cfg)
        n, h = cfg.n_heads, cfg.d_hidden

        self.refl = Attention(n, h, **kw)
        self.attn = Attention(n, h, **kw)

        self.lin1 = Linear(cfg.d_ffnet, **kw)
        self.activ = cfg.activation
        self.drop = Dropout()
        self.lin2 = Linear(h, **kw)

        self.norm1 = LayerNorm(h, cfg.norm_eps, **kw)
        self.norm2 = LayerNorm(h, cfg.norm_eps, **kw)
        self.norm3 = LayerNorm(h, cfg.norm_eps, **kw)
        self.drop1 = Dropout()
        self.drop2 = Dropout()
        self.drop3 = Dropout()

    def __setstate__(self, x):
        if "activation" not in x:
            x["activation"] = F.relu
        super(Decoder, self).__setstate__(x)

    def forward(self, tgt, mem, mask=None, k_mask=None):
        x = tgt
        if self.cfg.norm_first:
            x = x + self.reflect(self.norm1(x), tgt_mask, tgt_k_mask)
            x = x + self.attention(self.norm2(x), mem, mem_mask, mem_k_mask)
            x = x + self.ffnet(self.norm3(x))
        else:
            x = self.norm1(x + self.reflect(x, tgt_mask, tgt_k_mask))
            x = self.norm2(x + self.attention(x, mem, mem_mask, mem_k_mask))
            x = self.norm3(x + self.ffnet(x))
        return x

    def reflect(self, x, mask, k_mask):
        x = self.refl(x, x, x, attn_mask=mask, k_mask=k_mask, need_weights=False)[0]
        return self.drop1(x)

    def attention(self, x, mem, mask, k_mask):
        x = self.attn(x, mem, mem, attn_mask=mask, k_mask=k_mask, need_weights=False)[0]
        return self.drop2(x)

    def ffnet(self, x):
        x = self.lin2(self.drop(self.activ(self.lin1(x))))
        return self.drop3(x)


def _clones(m, n):
    return Stack([copy.deepcopy(m) for _ in range(n)])
