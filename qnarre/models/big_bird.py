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
# https://arxiv.org/abs/2007.14062
# https://github.com/google-research/bigbird

import numpy as np
import torch

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.big_bird import PreTrained

from . import bert

log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, add_pool=True, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(cfg.d_model, **kw)
        self.enc = Encoder(**kw)
        self.pool = Pool(**kw) if add_pool else None
        if cfg.attn_type != "original_full" and cfg.add_cross:
            self.set_attn_type("original_full")

    def set_attn_type(self, x):
        assert x in ["original_full", "block_sparse"]
        cfg = self.cfg
        if x == cfg.attn_type:
            return
        cfg.attn_type = x
        self.enc.set_attention_type(x)

    def forward(
        self,
        x,
        cache=None,
        enc_m=None,
        enc=None,
        head_m=None,
        mask=None,
        pos=None,
        typ=None,
        x_emb=None,
        **kw,
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x is not None:
            assert x_emb is None
            s, d = x.size(), x.device
        else:
            s, d = x_emb.size()[:-1], x_emb.device
        b, n = s
        c_len = cache[0][0].shape[2] if cache is not None else 0
        if mask is None:
            mask = torch.ones(((b, n + c_len)), device=d)
        if typ is None:
            if hasattr(self.embs, "typ_ids"):
                typ = self.embs.typ_ids[:, :n].expand(b, n)
            else:
                typ = torch.zeros(s, dtype=torch.long, device=d)
        max_tokens_to_attend = (5 + 2 * cfg.n_rand_blocks) * cfg.block_size
        if cfg.attn_type == "block_sparse" and n <= max_tokens_to_attend:
            n = x.size(1) if x is not None else x_emb.size(1)
            self.set_attn_type("original_full")
        if cfg.attn_type == "block_sparse":
            (p_len, x, mask, typ, pos, x_emb) = self.pad_to_block(
                x, mask=mask, pos=pos, typ=typ, x_emb=x_emb
            )
        else:
            p_len = 0
        if cfg.attn_type == "block_sparse":
            (blocked_enc_m, band_m, from_m, to_m) = self.create_masks_for_block(
                mask, self.block_size
            )
            mask = None
        else:
            assert cfg.attn_type == "original_full"
            blocked_enc_m = None
            band_m = None
            from_m = None
            to_m = None
            mask = self.get_mask(mask, s, d)
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        ys = self.embs(x, c_len=c_len, pos=pos, typ=typ, x_emb=x_emb)
        if not cfg.is_dec:
            yo.cache = False
        ys = self.enc(
            ys,
            band_m=band_m,
            blocked_enc_m=blocked_enc_m,
            cache=cache,
            enc_m=enc_m,
            enc=enc,
            from_m=from_m,
            head_m=head_m,
            mask=mask,
            to_m=to_m,
            yo=yo,
        )
        y = ys[0]
        pools = self.pool(y[:, 0, :]) if self.pool is not None else None
        if p_len > 0:
            y = y[:, :-p_len]
        ys = (y,) + ys[1:] + (pools,)
        return qo.PoolsCrosses(*ys) if yo.kw else ys

    @staticmethod
    def create_masks_for_block(mask, block):
        b, n = mask.size()
        assert n % block == 0

        def create_band_m(from_m, to_m):
            to_pad = torch.cat([to_m[:, 1:-3], to_m[:, 2:-2], to_m[:, 3:-1]], dim=2)
            y = torch.einsum("blq,blk->blqk", from_m[:, 2:-2], to_pad)
            y.unsqueeze_(1)
            return y

        enc_m = mask.view(b, n // block, block)
        band_m = create_band_m(enc_m, enc_m)
        from_m = mask.view(b, 1, n, 1)
        to_m = mask.view(b, 1, 1, n)
        return enc_m, band_m, from_m, to_m

    def pad_to_block(self, x, mask, typ, pos, x_emb, PAD):
        cfg = self.cfg
        block_size = cfg.block_size
        shape = x.shape if x is not None else x_emb.shape
        b, n = shape[:2]
        p_len = (block_size - n % block_size) % block_size
        if p_len > 0:
            if x is not None:
                x = F.pad(x, (0, p_len), value=PAD)
            if pos is not None:
                pos = F.pad(pos, (0, p_len), value=PAD)
            if x_emb is not None:
                p = x_emb.new_full((b, p_len), cfg.PAD, dtype=torch.long)
                x_emb = torch.cat([x_emb, self.embs(p)], dim=-2)
            mask = F.pad(mask, (0, p_len), value=False)
            typ = F.pad(typ, (0, p_len), value=0)
        return p_len, x, mask, typ, pos, x_emb


class ForCausal(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(**kw)

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        ys = self.model(x, yo=yo**kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            sl = y[:, :-1, :].contiguous()
            ls = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(sl.view(-1, cfg.s_vocab), ls.view(-1))
        ys = (y,) + ys[2:] + (loss,)
        return qo.LossCrosses(*ys) if yo.kw else ys


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(**kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.proj = qc.Linear(cfg.d_model, 1, **kw)

    forward = bert.ForMultiChoice.forward


class ForPreTraining(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=True, **kw)
        self.proj = Masker(**kw)
        self.seq = qc.Linear(cfg.d_model, 2, **kw)

    def forward(self, x, labels=None, ns_labels=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        ys = self.model(x, yo=yo**kw)
        y = self.proj(ys[0])
        orders = self.seq(ys[1])
        loss = None
        if labels is not None:
            f = nn.CrossEntropyLoss()
            loss = f(y.view(-1, cfg.s_vocab), labels.view(-1))
            if loss is not None:
                loss = loss + f(orders.view(-1, 2), ns_labels.view(-1))
        ys = (y, orders) + ys[2:] + (loss,)
        return bert.LossSeq(*ys) if yo.kw else ys


def prep_q_mask(q_lens, n):
    y = torch.arange(0, n).to(q_lens.device)
    y.unsqueeze_(0)
    y = y < q_lens
    return y


class ForQA(PreTrained):
    def __init__(self, add_pool=False, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=add_pool, **kw)
        self.drop = qc.Dropout(cfg.drop)
        self.ff = FFNet(cfg.act, cfg.drop, cfg.eps, cfg)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels)

    def forward(self, x, beg=None, end=None, q_lens=None, typ=None, x_emb=None, **kw):
        yo = self.get_y_opts(**kw)
        n = x.size(1) if x is not None else x_emb.size(1)
        if q_lens is None and x is not None:
            q_lens = torch.argmax(x.eq(self.SEP).int(), dim=-1) + 1
            q_lens.unsqueeze_(1)
        y_m = None
        if q_lens is not None:
            y_m = prep_q_mask(q_lens, n)
            if typ is None:
                typ = (~y_m).long()
            y_m[:, 0] = False
            y_m.unsqueeze_(2)
        ys = self.model(x, typ=typ, x_emb=x_emb, **kw, yo=yo)
        y = self.proj(self.ff(self.drop(ys[0])))
        if y_m is not None:
            y = y - y_m * 1e6
        b, e = y.split(1, dim=-1)
        b = b.squeeze(-1).contiguous()
        e = e.squeeze(-1).contiguous()
        loss = None
        if beg is not None and end is not None:
            if len(beg.size()) > 1:
                beg = beg.squeeze(-1)
            if len(end.size()) > 1:
                end = end.squeeze(-1)
            i = b.size(1)
            f = nn.CrossEntropyLoss(ignore_index=i)
            beg = beg.clamp(0, i)
            end = end.clamp(0, i)
            loss = (f(b, beg) + f(e, end)) / 2
        ys = (b, e) + ys[2:] + (loss,)
        return qo.LossQAPools(*ys) if yo.kw else ys


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, cfg.act, **kw)

    forward = qf.forward_seq  # y = self.proj(ys[0][:, 0, :])


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class Encoder(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([Layer(seed=i, **kw) for i in range(cfg.n_lays)])
        self.grad_checkpoint = False

    def set_attention_type(self, x):
        assert x in ["original_full", "block_sparse"]
        cfg = self.cfg
        if x == cfg.attn_type:
            return
        cfg.attn_type = x
        for lay in self.lays:
            lay.set_attention_type(x)

    def forward(self, x, cache=None, head_m=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        y = x
        attns = () if yo.attn else None
        caches = () if yo.cache else None
        crosses = () if yo.attn and cfg.add_cross else None
        hiddens = () if yo.hidden else None
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y,)
            h = head_m[i] if head_m is not None else None
            c = cache[i] if cache is not None else None
            if self.grad_checkpoint and self.training:
                if yo.cache:
                    yo.cache = False

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c, yo=yo)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw, head_m=h)
            else:
                ys = lay(y, **kw, cache=c, head_m=h, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
                if cfg.add_cross:
                    crosses += (ys[2],)
            if yo.cache:
                caches += (ys[-1],)
        if yo.hidden:
            hiddens += (y,)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(*ys) if yo.kw else ys


class Layer(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, seed=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        cfg.attn_type = cfg.attn_type
        self.attn = Attention(cfg, seed=seed)
        self.is_dec = cfg.is_dec
        self.add_cross = cfg.add_cross
        if self.add_cross:
            assert self.is_dec
            self.cross = Attention(cfg)
        self.ffnet = FFNet(cfg.act, cfg.drop, cfg.eps, **kw)

    def set_attention_type(self, x):
        assert x in ["original_full", "block_sparse"]
        cfg = self.cfg
        if x == cfg.attn_type:
            return
        cfg.attn_type = x
        self.attn.set_attention_type(x)
        if self.add_cross:
            self.cross.set_attention_type(x)

    def forward(
        self,
        x,
        mask=None,
        head_m=None,
        enc=None,
        enc_m=None,
        band_m=None,
        from_m=None,
        to_m=None,
        blocked_encoder_mask=None,
        prev_kv=None,
        y_attn=False,
        **kw,
    ):
        yo = self.get_y_opts(y_attn=y_attn, **kw)
        self_attn_past_key_value = prev_kv[:2] if prev_kv is not None else None
        self_attention_outputs = self.attn(
            x,
            mask,
            head_m,
            enc=enc,
            enc_m=enc_m,
            prev_kv=self_attn_past_key_value,
            band_m=band_m,
            from_m=from_m,
            to_m=to_m,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask,
            yo=yo,
        )
        attention_output = self_attention_outputs[0]
        if self.is_dec:
            y = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            y = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_dec and enc is not None:
            assert hasattr(self, "crossattention")
            cross_attn_past_key_value = prev_kv[-2:] if prev_kv is not None else None
            cross_attention_outputs = self.cross(
                attention_output,
                mask,
                head_m,
                enc,
                enc_m,
                cross_attn_past_key_value,
                yo=yo,
            )
            attention_output = cross_attention_outputs[0]
            y = y + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = self.ffnet(
            attention_output,
        )
        y = (layer_output,) + y
        if self.is_dec:
            y = y + (present_key_value,)
        return y


class Attention(qc.Module):
    hs = qc.Hypers({"d_embed", "d_model", "n_heads", "use_bias", "attn_type"}, {"drop_attn": 0.0})

    def __init__(self, seed=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        cfg.seed = seed
        if cfg.attn_type == "original_full":
            self.attn = FullAttn(**kw)
        else:
            assert cfg.attn_type == "block_sparse"
            self.attn = SparseAttn(seed, **kw)
        m = cfg.d_model
        self.proj = qc.Linear(m, m, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.norm = qc.LayerNorm(m, **kw)

    def set_attention_type(self, x):
        cfg = self.cfg
        assert x in ["original_full", "block_sparse"]
        if x == cfg.attn_type:
            return
        cfg.attn_type = x
        if x == "original_full":
            a = FullAttn(**kw)
        else:
            a = SparseAttn(cfg.seed, **kw)
        a.query = self.attn.query
        a.value = self.attn.value
        a.key = self.attn.key
        self.attn = a
        cfg.attn_type = x
        if not self.training:
            self.attn.eval()

    def forward(self, x, enc=None, **kw):
        yo = self.get_y_opts(**kw)
        if self.cfg.attn_type == "original_full":
            ys = self.attn(x, **kw, enc=enc, yo=yo)
        else:
            assert enc is None
            ys = self.attn(x, **kw, yo=yo)
        y = self.norm(x + self.drop(self.proj(ys[0])))
        y = (y,) + ys[1:]
        return y


class FullAttn(qc.Module):
    hs = qc.Hypers(
        {"d_embed", "d_model", "n_heads", "use_bias"},
        {"drop_attn": 0.0},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0 or cfg.d_embed is not None
        cfg.d_head = h = m // n
        cfg.scale = 1 / (h**0.5)
        self.query = qc.Linear(m, m, bias=cfg.use_bias, **kw)
        self.key = qc.Linear(m, m, bias=cfg.use_bias, **kw)
        self.value = qc.Linear(m, m, bias=cfg.use_bias, **kw)
        self.drop = qc.Dropout(cfg.drop_attn, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        q = self.split_heads(self.query(x))
        if enc is None:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
            if cache is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
        else:
            mask = enc_m
            if cache is None:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
            else:
                k = cache[0]
                v = cache[1]
        a = torch.matmul(q, k.transpose(-1, -2))
        a.mul_(cfg.scale)
        if mask is not None:
            a += mask
        a = self.drop(F.softmax(a, dim=-1))
        if head_m is not None:
            a *= head_m
        y = torch.matmul(a, v).permute(0, 2, 1, 3).contiguous()
        y = y.view(y.size()[:-2] + (cfg.n_heads * cfg.d_head,))
        y = (y,)
        if yo.attn:
            y += (a,)
        if yo.cache:
            y += ((k, v),)
        return y


class SparseAttn(qc.Module):
    hs = qc.Hypers(
        {"d_embed", "d_model", "n_heads", "n_pos", "use_bias", "n_rand_blocks", "block_size"},
        {"drop_attn": 0.0},
    )

    def __init__(self, seed=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        cfg.seed = seed
        m, n = cfg.d_model, cfg.n_heads
        assert m % n == 0
        cfg.d_head = int(m / n)
        cfg.s_all_head = h = n * cfg.d_head
        self.query = qc.Linear(m, h, bias=cfg.use_bias, **kw)
        self.key = qc.Linear(m, h, bias=cfg.use_bias, **kw)
        self.value = qc.Linear(m, h, bias=cfg.use_bias, **kw)

    split_heads = qa.split_heads

    def forward(self, x, **kw):
        yo = self.get_y_opts(**kw)
        b, seqlen, _ = x.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size
        assert from_seq_length % from_block_size == 0
        assert to_seq_length % to_block_size == 0
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        ctx, y = self.bigbird_block_sparse_attention(
            q,
            k,
            v,
            band_m,
            from_m,
            to_m,
            from_blocked_mask,
            to_blocked_mask,
            d_head,
            from_block_size,
            to_block_size,
            b,
            from_seq_length,
            to_seq_length,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            yo=yo,
            **kw,
        )
        ctx = ctx.contiguous().view(b, from_seq_length, -1)
        y = (ctx, y) if yo.attn else (ctx,)
        return y

    @staticmethod
    def torch_bmm_nd(x1, x2, ndim=None):
        s1, s2 = x1.shape, x2.shape
        return torch.bmm(x1.reshape((-1,) + s1[-2:]), x2.reshape((-1,) + s2[-2:])).view(
            s1[: ndim - 2] + (s1[ndim - 2], s2[ndim - 1]),
        )

    @staticmethod
    def torch_bmm_nd_transpose(x1, x2, ndim=None):
        s1, s2 = x1.shape, x2.shape
        return torch.bmm(
            x1.reshape((-1,) + s1[-2:]),
            x2.reshape((-1,) + s2[-2:]).transpose(1, 2),
        ).view(s1[: ndim - 2] + (s1[ndim - 2], s2[ndim - 2]))

    def bigbird_block_sparse_attention(
        self,
        q,
        k,
        v,
        band_m,
        from_m,
        to_m,
        from_blocked_mask,
        to_blocked_mask,
        d_head,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        plan_from_length,
        plan_num_rand_blocks,
        **kw,
    ):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        assert from_seq_len // from_block_size == to_seq_len // to_block_size
        rsqrt_d = 1 / (d_head**0.5)
        bsz = batch_size
        attn_mask_penalty = -10000.0
        np.random.seed(cfg.seed)
        if from_seq_len in [1024, 3072, 4096]:
            rand_attn = [
                self._bigbird_block_rand_mask(
                    cfg.n_pos,
                    cfg.n_pos,
                    from_block_size,
                    to_block_size,
                    last_idx=1024,
                )[: (from_seq_len // from_block_size - 2)]
                for _ in range(cfg.n_heads)
            ]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                    from_seq_len, from_block_size
                )
            rand_attn = self._bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_len,
                to_seq_length=to_seq_len,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )
        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.tensor(rand_attn, device=q.device, dtype=torch.long)
        rand_attn.unsqueeze_(0)
        rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask,
            to_blocked_mask,
            rand_attn,
            bsz,
            from_seq_len,
            from_block_size,
        )
        q = q.view(bsz, cfg.n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = k.view(
            bsz, cfg.n_heads, to_seq_len // to_block_size, to_block_size, -1
        )
        blocked_value_matrix = v.view(
            bsz, cfg.n_heads, to_seq_len // to_block_size, to_block_size, -1
        )
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(
            bsz, cfg.n_heads, to_seq_len // to_block_size - 2, cfg.n_rand_blocks * to_block_size, -1
        )
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(
            bsz, cfg.n_heads, to_seq_len // to_block_size - 2, cfg.n_rand_blocks * to_block_size, -1
        )
        first_product = self.torch_bmm_nd_transpose(q[:, :, 0], k, ndim=4)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_m) * attn_mask_penalty
        first_attn_weights = F.softmax(first_product, dim=-1)
        first_context_layer = self.torch_bmm_nd(first_attn_weights, v, ndim=4)
        first_context_layer.unsqueeze_(2)
        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )
        second_product = self.torch_bmm_nd_transpose(q[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_m[:, :, :, : 3 * to_block_size],
                to_m[:, :, :, -to_block_size:],
                to_m.new_ones([bsz, 1, 1, cfg.n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, cfg.n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, 0],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = F.softmax(second_product, dim=-1)
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)
        second_context_layer.unsqueeze_(2)
        exp_blocked_key_matrix = torch.cat(
            [
                blocked_key_matrix[:, :, 1:-3],
                blocked_key_matrix[:, :, 2:-2],
                blocked_key_matrix[:, :, 3:-1],
            ],
            dim=3,
        )
        exp_blocked_value_matrix = torch.cat(
            [
                blocked_value_matrix[:, :, 1:-3],
                blocked_value_matrix[:, :, 2:-2],
                blocked_value_matrix[:, :, 3:-1],
            ],
            dim=3,
        )
        middle_query_matrix = q[:, :, 2:-2]
        inner_band_product = self.torch_bmm_nd_transpose(
            middle_query_matrix, exp_blocked_key_matrix, ndim=5
        )
        inner_band_product = inner_band_product * rsqrt_d
        rand_band_product = self.torch_bmm_nd_transpose(
            middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5
        )
        rand_band_product = rand_band_product * rsqrt_d
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )
        first_band_product = first_band_product * rsqrt_d
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )
        last_band_product = last_band_product * rsqrt_d
        inner_band_product += (1.0 - band_m) * attn_mask_penalty
        first_band_product += (1.0 - to_m[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_m[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
        band_product = torch.cat(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1
        )
        attn_weights = F.softmax(band_product, dim=-1)
        ctx = self.torch_bmm_nd(
            attn_weights[:, :, :, :, to_block_size : 4 * to_block_size],
            exp_blocked_value_matrix,
            ndim=5,
        )
        ctx += self.torch_bmm_nd(
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size],
            gathered_value[:, :, 1:-1],
            ndim=5,
        )
        ctx += torch.einsum(
            "bhlqk,bhkd->bhlqd",
            attn_weights[:, :, :, :, :to_block_size],
            blocked_value_matrix[:, :, 0],
        )
        ctx += torch.einsum(
            "bhlqk,bhkd->bhlqd",
            attn_weights[:, :, :, :, -to_block_size:],
            blocked_value_matrix[:, :, -1],
        )
        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )
        second_last_product = self.torch_bmm_nd_transpose(q[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_m[:, :, :, :to_block_size],
                to_m[:, :, :, -3 * to_block_size :],
                to_m.new_ones([bsz, 1, 1, cfg.n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, cfg.n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, -1],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (
            1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)
        ) * attn_mask_penalty
        second_last_attn_weights = F.softmax(second_last_product, dim=-1)
        second_last_context_layer = self.torch_bmm_nd(
            second_last_attn_weights, second_last_value_mat, ndim=4
        )
        second_last_context_layer.unsqueeze_(2)
        last_product = self.torch_bmm_nd_transpose(q[:, :, -1], k, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_m) * attn_mask_penalty
        last_attn_weights = F.softmax(last_product, dim=-1)
        last_context_layer = self.torch_bmm_nd(last_attn_weights, v, ndim=4)
        last_context_layer.unsqueeze_(2)
        ctx = torch.cat(
            [
                first_context_layer,
                second_context_layer,
                ctx,
                second_last_context_layer,
                last_context_layer,
            ],
            dim=2,
        )
        ctx = ctx.view((bsz, cfg.n_heads, from_seq_len, -1)) * from_m
        ctx = torch.transpose(ctx, 1, 2)
        if yo.attn:
            y = torch.zeros(
                bsz,
                cfg.n_heads,
                from_seq_len,
                to_seq_len,
                dtype=torch.float,
                device=ctx.device,
            )
            y[:, :, :from_block_size, :] = first_attn_weights  # all keys global
            y[
                :, :, from_block_size : 2 * from_block_size, : 3 * to_block_size
            ] = second_attn_weights[:, :, :, : 3 * to_block_size]
            y[:, :, from_block_size : 2 * from_block_size, -to_block_size:] = second_attn_weights[
                :, :, :, 3 * to_block_size : 4 * to_block_size
            ]
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
                for p2, i2, w2 in zip(range(cfg.n_heads), i1, w1):
                    attn_probs_view = y.view(
                        bsz,
                        cfg.n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(
                        from_block_size, cfg.n_rand_blocks, to_block_size
                    )
            for i in range(from_seq_len // from_block_size - 4):
                attn_probs_view = y.view(
                    bsz,
                    cfg.n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, i, :, to_block_size : 4 * to_block_size]
                attn_probs_view[:, :, i, :, i : i + 3, :] = right_slice.view(
                    bsz, cfg.n_heads, from_block_size, 3, to_block_size
                )
            y[:, :, 2 * from_block_size : -2 * from_block_size, :to_block_size] = attn_weights[
                :, :, :, :, :to_block_size
            ].view(bsz, cfg.n_heads, -1, to_block_size)
            y[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
                :, :, :, :, -to_block_size:
            ].view(bsz, cfg.n_heads, -1, to_block_size)
            for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
                for p2, i2, w2 in zip(range(cfg.n_heads), i1, w1):
                    for i in range(1, len(i2) - 1):
                        attn_probs_view = y.view(
                            bsz,
                            cfg.n_heads,
                            from_seq_len // from_block_size,
                            from_block_size,
                            to_seq_len // to_block_size,
                            to_block_size,
                        )
                        right_slice = w2[i - 1, :, 4 * to_block_size : -to_block_size]
                        attn_probs_view[p1, p2, i + 1, :, i2[i]] = right_slice.view(
                            from_block_size, cfg.n_rand_blocks, to_block_size
                        )
            y[
                :, :, -2 * from_block_size : -from_block_size, :to_block_size
            ] = second_last_attn_weights[:, :, :, :to_block_size]
            y[
                :, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :
            ] = second_last_attn_weights[:, :, :, to_block_size : 4 * to_block_size]
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
                for p2, i2, w2 in zip(range(cfg.n_heads), i1, w1):
                    attn_probs_view = y.view(
                        bsz,
                        cfg.n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
                        from_block_size, cfg.n_rand_blocks, to_block_size
                    )
            y[:, :, -from_block_size:, :] = last_attn_weights
        else:
            y = None
        return ctx, y

    @staticmethod
    def torch_gather_b2(params, indices):
        assert params.shape[:2] == indices.shape[:2]
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]
        indices_shift = (
            torch.arange(
                indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device
            )
            // num_indices_to_gather
            * num_indices_to_pick_from
        )
        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])
        y = flattened_params.index_select(0, flattened_indices)
        y = y.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return y

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        rand_attn,
        num_rand_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(
            batch_size, n_heads, num_windows, num_rand_blocks * from_block_size
        )
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)
        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        assert from_seq_length // from_block_size == to_seq_length // to_block_size
        rand_attn = np.zeros(
            (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32
        )
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1
        r = num_rand_blocks
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        cfg = self.cfg
        assert from_seq_length // from_block_size == to_seq_length // to_block_size
        assert from_seq_length in plan_from_length
        num_blocks = from_seq_length // from_block_size
        plan_block_length = np.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [
            np.zeros((num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])), dtype=np.int32)
            for i in range(cfg.n_heads)
        ]
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(cfg.n_heads):
                            rand_attn[h][
                                blk_rw_idx, rnd_r_cnt:curr_r_cnt
                            ] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )
                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(
                        plan_block_length[plan_idx - 1], plan_block_length[plan_idx]
                    ):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(cfg.n_heads):
                            rand_attn[h][
                                blk_rw_idx, rnd_r_cnt:curr_r_cnt
                            ] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(cfg.n_heads):
                    rand_attn[h][
                        blk_rw_idx, rnd_r_cnt:curr_r_cnt
                    ] = self._get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )
        for nh in range(cfg.n_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]
        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        perm_block = np.random.permutation(to_block_list)
        illegal_blocks = list(
            range(block_id - window_block_left, block_id + window_block_right + 1)
        )
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)
        selected_random_blokcs = []
        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)
