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
# https://arxiv.org/abs/1910.10683
# https://github.com/google-research/text-to-text-transfer-transformer

import math
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.utils import logging

# from apex.normalization import FusedRMSNorm
# LayerNorm = FusedRMSNorm

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..prep.config.t5 import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embed = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        kw.update(is_dec=False, is_enc_dec=False)
        self.enc = Encoder(self.embed, **kw)
        kw.update(is_dec=True, is_enc_dec=False, n_lays=cfg.n_dec_lays)
        self.dec = Encoder(self.embed, **kw)

    def forward(
        self,
        x,
        dec_head_m=None,
        dec_m=None,
        head_m=None,
        mask=None,
        x_dec_emb=None,
        x_dec=None,
        x_emb=None,
        y_enc=None,
        **kw,
    ):
        cfg = self.cfg
        if head_m is not None and dec_head_m is None:
            if cfg.n_lays == cfg.n_dec_lays:
                dec_head_m = head_m
        if y_enc is None:
            y_enc = self.enc(x, mask=mask, x_emb=x_emb, head_m=head_m, **kw)
        y = self.dec(
            x_dec,
            **kw,
            enc_m=mask,
            enc=y_enc[0],
            head_m=dec_head_m,
            mask=dec_m,
            x_emb=x_dec_emb,
        )
        ys = y + y_enc
        return qo.Seq2Seq(*ys)


class ForCondGen(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embed = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        kw.update(is_dec=False, is_enc_dec=False)
        self.enc = Encoder(self.embed, **kw)
        kw.update(is_dec=True, is_enc_dec=False, n_lays=cfg.n_dec_lays)
        self.dec = Encoder(self.embed, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, bias=False, **kw)

    def forward(
        self,
        x,
        dec_head_m=None,
        dec_m=None,
        head_m=None,
        labels=None,
        mask=None,
        x_dec_emb=None,
        x_dec=None,
        y_enc=None,
        **kw,
    ):
        cfg = self.cfg
        if head_m is not None and dec_head_m is None:
            if cfg.n_lays == cfg.n_dec_lays:
                dec_head_m = head_m
        if y_enc is None:
            y_enc = self.enc(x, mask=mask, head_m=head_m, **kw)
        if labels is not None and x_dec is None and x_dec_emb is None:
            x_dec = self._shift_right(labels)
        ys = self.dec(
            x_dec,
            **kw,
            enc_m=mask,
            enc=y_enc[0],
            head_m=dec_head_m,
            mask=dec_m,
            x_emb=x_dec_emb,
        )
        y = ys[0]
        if cfg.tie_word_embeds:
            y = y * (cfg.d_model**-0.5)
        y = self.proj(y)
        loss = None
        if labels is not None:
            f = nn.CrossEntropyLoss(ignore_index=-100)
            loss = f(y.view(-1, y.size(-1)), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossSeq2Seq(*ys)


class LayerNorm(qc.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + self.variance_eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            y = y.to(self.weight.dtype)
        return self.weight * y


class DenseReluDense(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wi = qc.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = qc.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = qc.Dropout(cfg.drop_rate)

    def forward(self, x):
        y = self.wi(x)
        y = F.relu(y)
        y = self.drop(y)
        y = self.wo(y)
        return y


class DenseGatedGeluDense(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wi_0 = qc.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = qc.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = qc.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = qc.Dropout(cfg.drop_rate)
        self.act = qu.activation("gelu_new")

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        y = hidden_gelu * hidden_linear
        y = self.drop(y)
        y = self.wo(y)
        return y


class LayerFF(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.feed_forward_proj == "relu":
            self.DenseReluDense = DenseReluDense(cfg)
        elif cfg.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = DenseGatedGeluDense(cfg)
        else:
            raise ValueError(
                f"{cfg.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        self.norm = LayerNorm(cfg.d_model, cfg.eps)
        self.drop = qc.Dropout(cfg.drop_rate)

    def forward(self, x):
        y = self.norm(x)
        y = self.DenseReluDense(y)
        y = x + self.drop(y)
        return y


class Attention(qc.Module):
    def __init__(self, cfg, has_relative_attention_bias=False):
        super().__init__()
        self.is_dec = cfg.is_dec
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = cfg.relative_attention_num_buckets
        cfg.d_model = cfg.d_model
        self.key_value_proj_dim = cfg.d_kv
        cfg.n_heads = cfg.n_heads
        self.drop = cfg.drop_rate
        self.inner_dim = cfg.n_heads * self.key_value_proj_dim
        self.q = qc.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.k = qc.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.v = qc.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.o = qc.Linear(self.inner_dim, cfg.d_model, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = qc.Embed(
                self.relative_attention_num_buckets, cfg.n_heads
            )
        self.grad_checkpoint = False

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_dec),
            num_buckets=self.relative_attention_num_buckets,
        )
        y = self.relative_attention_bias(relative_position_bucket)
        y = y.permute([2, 0, 1]).unsqueeze(0)
        return y

    def forward(
        self, x, mask=None, kv=None, pos=None, prev_kv=None, head_m=None, query_length=None, **kw
    ):
        cfg = self.cfg
        b, seq_length = x.shape[:2]
        real_seq_length = seq_length
        if prev_kv is not None:
            assert len(prev_kv) == 2
            real_seq_length += prev_kv[0].shape[2] if query_length is None else query_length
        key_length = real_seq_length if kv is None else kv.shape[1]

        def shape(x):
            return x.view(b, -1, cfg.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(b, -1, self.inner_dim)

        def project(x, proj_layer, prev_kv):
            if kv is None:
                x = shape(proj_layer(x))
            elif prev_kv is None:
                x = shape(proj_layer(kv))
            if prev_kv is not None:
                if kv is None:
                    x = torch.cat([prev_kv, x], dim=2)
                else:
                    x = prev_kv
            return x

        q = shape(self.q(x))
        k = project(x, self.k, prev_kv[0] if prev_kv is not None else None)
        v = project(x, self.v, prev_kv[1] if prev_kv is not None else None)
        y = torch.matmul(q, k.transpose(3, 2))
        if pos is None:
            if not self.has_relative_attention_bias:
                pos = torch.zeros(
                    (1, cfg.n_heads, real_seq_length, key_length),
                    device=y.device,
                    dtype=y.dtype,
                )
                if self.grad_checkpoint and self.training:
                    pos.requires_grad = True
            else:
                pos = self.compute_bias(real_seq_length, key_length)
            if prev_kv is not None:
                pos = pos[:, :, -x.size(1) :, :]
            if mask is not None:
                pos = pos + mask
        y += pos
        a = F.softmax(y.float(), dim=-1).type_as(y)
        a = self.drop(a)
        if head_m is not None:
            a = a * head_m
        y = unshape(torch.matmul(a, v))
        y = self.o(y)
        kv = (k, v) if (self.is_dec) else None
        return y, kv, pos, a


class Reflection(qc.Module):
    def __init__(self, cfg, has_relative_attention_bias=False):
        super().__init__()
        self.refl = Attention(cfg, has_relative_attention_bias=has_relative_attention_bias)
        self.norm = LayerNorm(cfg.d_model, cfg.eps)
        self.drop = qc.Dropout(cfg.drop_rate)

    def forward(self, x, **kw):
        x = self.norm(x)
        ys = self.refl(x, **kw)
        x = x + self.drop(ys[0])
        y = (x,) + ys[1:]
        return y


class Cross(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg, has_relative_attention_bias=False)
        self.norm = LayerNorm(cfg.d_model, cfg.eps)
        self.drop = qc.Dropout(cfg.drop_rate)

    def forward(self, x, key_value_states, **kw):
        x = self.norm(x)
        ys = self.attn(x, key_value_states=key_value_states, **kw)
        y = x + self.drop(ys[0])
        y = (y,) + ys[1:]
        return y


class Block(qc.Module):
    def __init__(self, cfg, has_relative_attention_bias=False):
        super().__init__()
        self.is_dec = cfg.is_dec
        self.lays = qc.Stack()
        self.lays.append(Reflection(cfg, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_dec:
            self.lays.append(Cross(cfg))
        self.lays.append(LayerFF(cfg))

    def forward(
        self,
        y,
        enc=None,
        enc_m=None,
        encoder_decoder_position_bias=None,
        cross_m=None,
        prev_kv=None,
        **kw,
    ):
        cfg = self.cfg
        if prev_kv is not None:
            assert self.is_dec
            expected_num_past_key_values = 2 if enc is None else 4
            assert len(prev_kv) == expected_num_past_key_values
            pkv = prev_kv[:2]
            pkv2 = prev_kv[2:]
        else:
            pkv, pkv2 = None, None
        ys = self.lays[0](y, prev_kv=pkv, **kw)
        y, kv = ys[:2]
        ys = ys[2:]
        if y.dtype == torch.float16 and torch.isinf(y).any():
            clamp = torch.finfo(y.dtype).max - 1000
            y = torch.clamp(y, min=-clamp, max=clamp)
        if self.is_dec and enc is not None:
            if kv is not None:
                query_length = kv[0].shape[2]
            else:
                query_length = None
            ys2 = self.lays[1](
                y,
                **kw,
                enc=enc,
                mask=enc_m,
                position_bias=encoder_decoder_position_bias,
                head_m=cross_m,
                prev_kv=pkv2,
                query_length=query_length,
            )
            y = ys2[0]
            if y.dtype == torch.float16 and torch.isinf(y).any():
                clamp = torch.finfo(y.dtype).max - 1000
                y = torch.clamp(y, min=-clamp, max=clamp)
            if kv is not None:
                kv = kv + ys2[1]
            ys = ys + ys2[2:]
        y = self.lays[-1](y)
        if y.dtype == torch.float16 and torch.isinf(y).any():
            clamp = torch.finfo(y.dtype).max - 1000
            y = torch.clamp(y, min=-clamp, max=clamp)
        return y + (kv,) + ys


class Encoder(qc.Module):
    hs = qc.Hypers({"add_cross", "n_lays"})

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        kw.update(y_cache=False, is_enc_dec=False)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.tok_emb = qc.Embed(cfg.s_vocab, m, **kw) if tok_emb is None else tok_emb
        self.lays = qc.Stack(
            [Block(**kw, has_relative_attention_bias=bool(i == 0)) for i in range(cfg.n_lays)]
        )
        self.norm = LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop_rate, **kw)

    def forward(
        self,
        x,
        cache=None,
        cross_m=None,
        enc_m=None,
        enc=None,
        head_m=None,
        mask=None,
        x_emb=None,
        **kw,
    ):
        cfg = self.cfg
        if x is None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x)
        b, n = s
        mask_seq_length = cache[0][0].shape[2] + n if cache is not None else n
        if mask is None:
            mask = torch.ones(b, mask_seq_length).to(x_emb.device)
        mask = self.get_mask(mask, s, x_emb.device)
        if cfg.is_dec and enc_m is None and enc is not None:
            enc_m = torch.ones(b, enc.shape[1], device=x_emb.device, dtype=torch.long)
        if cache is None:
            cache = tuple([None] * len(self.lays))
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=x_emb.device)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        cross_m = self.get_head_m(cross_m, cfg.n_lays)
        attns = caches = crosses = hiddens = ()
        pos = None
        enc_dec_pos = None
        y = self.drop(x_emb)
        for i, (lay, c) in enumerate(zip(self.lays, cache)):
            hiddens += (y,)
            kw.update(
                cross_m=cross_m[i],
                enc_dec_pos=enc_dec_pos,
                enc_m=enc_m,
                enc=enc,
                head_m=head_m[i],
                mask=mask,
                pos=pos,
            )
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw)
            else:
                ys = lay(y, cache=c, **kw)
            y, kv = ys[:2]
            pos = ys[2]
            if self.is_dec and enc is not None:
                enc_dec_pos = ys[4]
            attns += (ys[3],)
            if self.is_dec:
                crosses += (ys[5],)
            caches += (kv,)
        y = self.drop(self.norm(y))
        hiddens += (y,)
        return qo.CachesCrosses(y, attns, caches, crosses, hiddens)
