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

import random
import torch
import deepspeed

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import SinEmbed
from ..prep.config.fsmt import PreTrained


logger = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.enc = Encoder(**kw)
        self.dec = Decoder(**kw)

    def forward(self, x, mask=None, x_dec=None, dec_m=None, dec_head_m=None, y_enc=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x_dec is None:
            yo.cache = False
        if not yo.cache:
            x_dec, dec_m, causal_m = _prepare_fsmt_decoder_inputs(
                cfg,
                x,
                x_dec=x_dec,
                dec_m=dec_m,
                causal_m_dtype=self.dec.tok_emb.weight.dtype,
            )
        else:
            dec_m, causal_m = None, None
        assert x_dec is not None
        if y_enc is None:
            y_enc = self.enc(x, **kw, mask=mask, yo=yo)
        y = self.dec(
            x_dec,
            **kw,
            dec_causal_m=causal_m,
            enc_m=mask,
            enc=y_enc[0],
            head_m=dec_head_m,
            mask=dec_m,
            yo=yo,
        )
        ys = y + y_enc
        return qo.Seq2Seq(*ys) if yo.kw else ys


class ForCondGen(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if labels is not None:
            yo.cache = False
        ys = self.model(x, **kw, yo=yo)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(ys[0].view(-1, cfg.s_tgt_vocab), labels.view(-1))
        ys += (loss,)
        return qo.LossSeq2Seq(*ys) if yo.kw else ys


class Encoder(qc.Module):
    hs = qc.Hypers({"d_model", "drop", "n_heads", "n_pos"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_src_vocab, m, **kw)
        self.pos_emb = SinEmbed(cfg.n_pos + cfg.PAD + 1, m, cfg.PAD)
        self.lays = qc.Stack([EncLayer(**kw) for _ in range(cfg.n_enc_lays)])
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, mask=None, head_m=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if mask is not None:
            mask = invert_mask(mask)
        y = self.tok_emb(x) * cfg.scale
        y = y + self.pos_emb(x)
        y = self.drop(y).transpose(0, 1)
        attns = () if yo.attn else None
        hiddens = () if yo.hidden else None
        assert head_m is None or (head_m.size()[0] == (len(self.lays)))
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y.transpose(0, 1),)
            if self.training and (random.uniform(0, 1) < cfg.drop_enc):
                continue
            else:
                h = head_m[i] if head_m is not None else None
                ys = lay(y, mask=mask, head_m=h, **kw, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
        y = y.transpose(0, 1)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, hiddens)
        return qo.Base(*ys) if yo.kw else ys


class Decoder(qc.Module):
    hs = qc.Hypers({"d_model", "drop", "n_heads", "n_pos"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        cfg.scale = m**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_tgt_vocab, m, **kw)
        self.pos_emb = SinEmbed(cfg.n_pos + cfg.PAD + 1, m, cfg.PAD)
        self.lays = qc.Stack([DecLayer(**kw) for _ in range(cfg.n_dec_lays)])
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.GatheredParameters(self.tok_emb.weight, modifier_rank=None):
                s = self.tok_emb.weight.shape
        else:
            s = self.tok_emb.weight.shape
        self.proj = qc.Linear(s[1], s[0], bias=False, **kw)
        self.proj.weight = self.tok_emb.weight
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(
        self, x, enc, enc_m, dec_m, dec_causal_m, head_m=None, cross_m=None, cache=None, **kw
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if enc_m is not None:
            enc_m = invert_mask(enc_m)
        y = self.tok_emb(x) * cfg.scale
        pos = self.pos_emb(x)
        if yo.cache:
            x = x[:, -1:]
            pos = pos[:, -1:]
        y += pos
        y = self.drop(y).transpose(0, 1)
        attns = () if yo.attn else None
        caches = () if yo.cache else None
        crosses = () if yo.attn else None
        hiddens = () if yo.hidden else None
        enc = enc.transpose(0, 1)
        for m, _ in zip([head_m, cross_m], ["head_m", "cross_m"]):
            if m is not None:
                assert m.size()[0] == (len(self.lays))
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y.transpose(0, 1),)
            if self.training and (random.uniform(0, 1) < cfg.drop_dec):
                continue
            h = head_m[i] if head_m is not None else None
            c = cross_m[i] if cross_m is not None else None
            kw.update(enc=enc, enc_m=enc_m, dec_m=dec_m, head_m=h, cross_m=c)
            c = cache[i] if cache is not None else None
            ys = lay(y, causal_m=dec_causal_m, cache=c, **kw, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
                if enc is not None:
                    crosses += (ys[2],)
            if yo.cache:
                caches += (ys[-1],)
        enc = enc.transpose(0, 1)
        y = y.transpose(0, 1)
        if yo.hidden:
            hiddens += (y,)
        y = self.proj(y)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys


class EncLayer(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_enc_heads, **kw)
        self.norm_refl = qc.LayerNorm(m, **kw)
        self.act = qu.activation(cfg.act_fun)
        self.ff = qc.Linear(m, cfg.d_enc_ffn, **kw)
        self.proj = qc.Linear(cfg.d_enc_ffn, m, **kw)
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, mask, **kw):
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, yo=None)
        y, a = self.refl(query=x, key=x, key_m=mask, **kw)
        y = self.norm_refl(x + self.drop(y))
        x = y
        y = self.drop(self.act(self.ff(y)))
        y = self.drop(self.proj(y))
        y = self.norm(x + y)
        y = (y,)
        if yo.attn:
            y += (a,)
        return y


class DecLayer(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_dec_heads, **kw)
        self.norm_refl = qc.LayerNorm(m, **kw)
        self.act = qu.activation(cfg.act_fun)
        self.drop_act = qc.Dropout(cfg.drop_act, **kw)
        self.attn = Attention(n_heads=cfg.n_dec_heads, enc_dec_attn=True, **kw)
        self.norm_attn = qc.LayerNorm(m, **kw)
        self.ff = qc.Linear(m, cfg.d_dec_ffn, **kw)
        self.proj = qc.Linear(cfg.d_dec_ffn, m, **kw)
        self.norm = qc.LayerNorm(m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(
        self, x, enc, enc_m=None, cache=None, causal_m=None, cross_m=None, dec_m=None, **kw
    ):
        yo = self.get_y_opts(**kw)
        kw.update(y_attn=True, y_cache=True, yo=None)
        if cache is None:
            cache = {}
        y, a = self.refl(query=x, key=x, key_m=dec_m, cache=cache, **kw, mask=causal_m)
        y = self.norm_refl(x + self.drop(y))
        x = y
        assert self.attn.cache_key != self.refl.cache_key
        y, kv = self.attn(query=y, key=enc, key_m=enc_m, cache=cache, **kw, head_m=cross_m)
        y = self.norm_attn(x + self.drop(y))
        x = y
        y = self.drop_act(self.act(self.ff(y)))
        y = self.drop(self.proj(y))
        y = self.norm(x + y)
        y = (y,)
        if yo.attn:
            y += (a, cache)
        if yo.cache:
            y += (kv,)
        return y


def invert_mask(mask):
    assert mask.dim() == 2
    return mask.eq(0)


def triu_onnx(x, diagonal=0):
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)


def _prepare_fsmt_decoder_inputs(
    config,
    input_ids,
    x_dec=None,
    decoder_padding_mask=None,
    causal_mask_dtype=torch.float32,
):
    PAD = config.PAD
    if x_dec is None:
        x_dec = qu.shift_right2(input_ids, PAD)
    bsz, tgt_len = x_dec.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(x_dec, PAD)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = triu_onnx(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=x_dec.device
    )
    return x_dec, decoder_padding_mask, causal_mask


def make_padding_mask(x, PAD=1):
    y = x.eq(PAD)
    if not y.any():
        y = None
    return y


class Attention(qc.Module):
    hs = qc.Hypers({"d_in", "d_out"}, {"drop": 0.0, "enc_dec_attn": False})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0
        cfg.d_head = h = m // n
        cfg.scale = 1 / (h**0.5)
        self.key = qc.Linear(m, m, **kw)
        self.value = qc.Linear(m, m, **kw)
        self.query = qc.Linear(m, m, **kw)
        self.proj = qc.Linear(m, m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.cache_key = "encoder_decoder" if self.enc_dec_attn else "self"

    split_heads = qa.split_heads

    def forward(self, x, mask=None, head_m=None, enc=None, enc_m=None, cache=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        static_kv = self.enc_dec_attn
        if cache is not None:
            saved_state = None
            cache = {}
        else:
            saved_state = cache.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                enc = None
        q = self.split_heads(self.query(x) * cfg.scale)
        if static_kv:
            if enc is None:
                k = v = None
            else:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
        else:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
        if saved_state is not None:
            k, v, enc_m = self._use_saved_state(k, v, saved_state, enc_m, static_kv, b)
        cache[self.cache_key] = {
            "prev_key": k.view(b, n, -1, cfg.d_head),
            "prev_value": v.view(b, n, -1, cfg.d_head),
            "prev_key_padding_mask": enc_m if not static_kv else None,
        }
        n = cfg.n_heads
        tgt, b, _ = x.size()
        src = k.size(1)
        y = torch.bmm(q, k.transpose(1, 2))
        assert y.size() == (b * n, tgt, src)
        if mask is not None:
            y = y.view(b, n, tgt, src) + mask
            y = y.view(b * n, tgt, src)
        if enc_m is not None and enc_m.dim() == 0:
            enc_m = None
        assert enc_m is None or enc_m.size()[:2] == (b, src)
        if enc_m is not None:
            y = y.view(b, n, tgt, src)
            reshaped = enc_m.unsqueeze(1).unsqueeze(2)
            y = y.masked_fill(reshaped, float("-inf"))
            y = y.view(b * n, tgt, src)
        y = F.softmax(y, dim=-1)
        if head_m is not None:
            assert head_m.size() == (n,)
            y = head_m.view(1, -1, 1, 1) * y.view(b, n, tgt, src)
            y = y.view(b * n, tgt, src)
        if yo.attn:
            a = y.view(b, n, tgt, src)
            y = a.view(b * n, tgt, src)
        y = self.drop(y)
        y = torch.bmm(y, v)
        assert y.size() == (b * n, tgt, cfg.d_head)
        y = y.transpose(0, 1).contiguous().view(tgt, b, cfg.d_model)
        y = self.proj(y)
        y = (y,)
        if yo.attn:
            y += (a,)
        return y

    def _use_saved_state(self, k, v, saved_state, key_m, static_kv, bsz):
        cfg = self.cfg
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * cfg.n_heads, -1, cfg.d_head)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * cfg.n_heads, -1, cfg.d_head)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_m], dim=1)
        else:
            new_key_padding_mask = key_m
        return k, v, new_key_padding_mask


def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)
