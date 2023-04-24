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
# https://arxiv.org/abs/1906.08237
# https://github.com/zihangdai/xlnet

import torch

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core.embed import pos_enc
from ..core.mlp import Classifier, FFNet, PoolBeg, PoolEnd, PoolProj
from ..prep.config.xlnet import PreTrained


log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embed = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        self.mask = nn.Parameter(torch.FloatTensor(1, 1, cfg.d_model))
        self.lays = qc.Stack([Layer(**kw) for _ in range(cfg.n_lays)])
        self.drop = qc.Dropout(cfg.drop, **kw)

    def create_mask(self, qlen, mlen):
        mask = torch.ones([qlen, qlen])
        up = torch.triu(mask, diagonal=1)
        pad = torch.zeros([qlen, mlen])
        y = torch.cat([pad, up], dim=1)
        if self.same_length:
            lo = torch.tril(mask, diagonal=-1)
            y = torch.cat([y[:, :qlen] + lo, y[:, qlen:]], dim=1)
        y = y.to(self.device)
        return y

    def cache_mem(self, x, prev):
        if self.reuse_len is not None and self.reuse_len > 0:
            x = x[: self.reuse_len]
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len
        if prev is None:
            y = x[cutoff:]
        else:
            y = torch.cat([prev, x], dim=0)[cutoff:]
        return y.detach()

    def forward(
        self,
        x,
        mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        typ=None,
        x_m=None,
        head_m=None,
        x_emb=None,
        use_mems=None,
        **kw,
    ):
        cfg = self.cfg
        if "y_cache" in kw:
            use_mems = kw["y_cache"]
        if self.training:
            use_mems = use_mems if use_mems is not None else cfg.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else cfg.use_mems_eval
        if x is not None:
            assert x_emb is None
            x = x.transpose(0, 1).contiguous()
            shape = x.size()
        else:
            x_emb = x_emb.transpose(0, 1).contiguous()
            shape = x_emb.size()[:-1]
        n, b = shape
        typ = typ.transpose(0, 1).contiguous() if typ is not None else None
        x_m = x_m.transpose(0, 1).contiguous() if x_m is not None else None
        mask = mask.transpose(0, 1).contiguous() if mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = (
            target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None
        )
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + n
        dtype_float = cfg.dtype
        device = cfg.device
        if cfg.attn_type == "uni":
            attn_mask = self.create_mask(n, mlen)
            attn_mask = attn_mask[:, :, None, None]
        else:
            assert cfg.attn_type == "bi"
            attn_mask = None
        assert x_m is None or mask is None
        if x_m is None and mask is not None:
            x_m = 1.0 - mask
        if x_m is not None and perm_mask is not None:
            data_mask = x_m[None] + perm_mask
        elif x_m is not None and perm_mask is None:
            data_mask = x_m[None]
        elif x_m is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None
        if data_mask is not None:
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, b]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]
        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)
        if attn_mask is not None:
            non_tgt_mask = -torch.eye(n).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat(
                    [torch.zeros([n, mlen]).to(attn_mask), non_tgt_mask], dim=-1
                )
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None
        if x_emb is not None:
            word_emb_k = x_emb
        else:
            word_emb_k = self.embed(x)
        output_h = self.drop(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask.expand(target_mapping.shape[0], b, -1)
            output_g = self.drop(word_emb_q)
        else:
            output_g = None
        if typ is not None:
            if mlen > 0:
                mem_pad = torch.zeros([mlen, b], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, typ], dim=0)
            else:
                cat_ids = typ
            seg_mat = (typ[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None
        pos = self.drop(pos_enc(n, klen, bsz=b))
        if head_m is not None:
            if head_m.dim() == 1:
                head_m = head_m.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_m = head_m.expand(cfg.n_lays, -1, -1, -1, -1)
            elif head_m.dim() == 2:
                head_m = head_m.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_m = head_m.to(dtype=next(self.parameters()).dtype)
        else:
            head_m = [None] * cfg.n_lays
        new_mems = ()
        if mems is None:
            mems = [None] * len(self.lays)
        attns = hiddens = ()
        for i, lay in enumerate(self.lays):
            if use_mems:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            hiddens += (output_h, output_g) if output_g is not None else (output_h,)
            ys = lay(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_m=head_m[i],
                **kw,
            )
            output_h, output_g = ys[:2]
            attns += (ys[2],)
        hiddens += (output_h, output_g) if output_g is not None else (output_h,)
        y = self.drop(output_g if output_g is not None else output_h)
        y = y.permute(1, 0, 2).contiguous()
        if not use_mems:
            new_mems = None
        if output_g is not None:
            hiddens = tuple(h.permute(1, 0, 2).contiguous() for hs in hiddens for h in hs)
        else:
            hiddens = tuple(hs.permute(1, 0, 2).contiguous() for hs in hiddens)
        if target_mapping is not None:
            attns = tuple(
                tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t)
                for t in attns
            )
        else:
            attns = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attns)
        return Output(y, attns, hiddens, new_mems)


@dataclass
class Output(qc.Output):
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None


class ForMultiChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.seqs = qc.SeqSummary(**kw)
        self.proj = qc.Linear(cfg.d_model, 1, **kw)

    def forward(self, x, typ=None, x_m=None, mask=None, x_emb=None, labels=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, typ, x_m, mask = qu.view_2D(x, typ, x_m, mask)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, typ=typ, x_m=x_m, mask=mask, x_emb=x_emb, **kw)
        y = self.proj(self.seqs(ys[0])).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return WithLoss(*ys)


@dataclass
class WithLoss(qc.Output):
    logits: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None
    loss: tuple = None


class ForQASimple(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    def forward(self, x, beg_pos=None, end_pos=None, **kw):
        ys = self.model(x, **kw)
        b, e = self.proj(ys[0]).split(1, dim=-1)
        b = b.squeeze(-1).contiguous()
        e = e.squeeze(-1).contiguous()
        loss = None
        if beg_pos is not None and end_pos is not None:
            if len(beg_pos.size()) > 1:
                beg_pos = beg_pos.squeeze(-1)
            if len(end_pos.size()) > 1:
                end_pos = end_pos.squeeze(-1)
            i = b.size(1)
            f = nn.CrossEntropyLoss(ignore_index=i)
            beg_pos = beg_pos.clamp(0, i)
            end_pos = end_pos.clamp(0, i)
            loss = (f(b, beg_pos) + f(e, end_pos)) / 2
        ys = (b, e) + ys[1:] + (loss,)
        return QA(*ys)


@dataclass
class QA(qc.Output):
    logits_beg: tuple = None
    logits_end: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None
    loss: tuple = None


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.logits_beg = PoolBeg(**kw)
        self.logits_end = PoolEnd(**kw)
        self.proj = PoolProj(**kw)

    def forward(
        self,
        x,
        beg_pos=None,
        end_pos=None,
        is_impossible=None,
        cls_index=None,
        p_mask=None,
        **kw,
    ):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = ys[0]
        s = self.logits_beg(y, p_mask=p_mask)
        if beg_pos is not None and end_pos is not None:
            for i in (beg_pos, end_pos, cls_index, is_impossible):
                if i is not None and i.dim() > 1:
                    i.squeeze_(-1)
            e = self.logits_end(y, beg_pos=beg_pos, p_mask=p_mask)
            f = nn.CrossEntropyLoss()
            loss = (f(s, beg_pos) + f(e, end_pos)) / 2
            if cls_index is not None and is_impossible is not None:
                y = self.proj(y, beg_pos=beg_pos, cls_index=cls_index)
                loss += nn.BCEWithLogitsLoss()(y, is_impossible) * 0.5
            ys = (y,) + ys[1:] + (loss,)
            return QATop(*ys)
        else:
            _, n, hsz = y.size()
            slps = F.softmax(s, dim=-1)
            top_beg, top_beg_i = torch.topk(slps, cfg.beg_n_top, dim=-1)
            x = top_beg_i.unsqueeze(-1).expand(-1, -1, hsz)
            ss = torch.gather(y, -2, x).unsqueeze(1).expand(-1, n, -1, -1)
            x = y.unsqueeze(2).expand_as(ss)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            e = self.logits_end(x, beg_states=ss, p_mask=p_mask)
            elps = F.softmax(e, dim=1)
            top_end, top_end_i = torch.topk(elps, cfg.end_n_top, dim=1)
            top_end = top_end.view(-1, cfg.beg_n_top * cfg.end_n_top)
            top_end_i = top_end_i.view(-1, cfg.beg_n_top * cfg.end_n_top)
            ss = torch.einsum("blh,bl->bh", y, slps)
            y = self.proj(y, beg_states=ss, cls_index=cls_index)
            ys = (y,) + ys[1:] + (top_beg, top_beg_i, top_end, top_end_i)
            return QATop(*ys)


@dataclass
class QATop(qc.Output):
    logits: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None
    top_beg = None
    top_beg_i = None
    top_end = None
    top_end_i = None
    loss: tuple = None


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.seqs = qc.SeqSummary(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = self.proj(self.seqs(ys[0]))
        loss = None
        if labels is not None:
            if cfg.problem is None:
                dt = labels.dtype
                if cfg.n_labels == 1:
                    cfg.problem = "regression"
                elif cfg.n_labels > 1 and (dt == torch.long or dt == torch.int):
                    cfg.problem = "single_label"
                else:
                    cfg.problem = "multi_label"
            if cfg.problem == "regression":
                if cfg.n_labels == 1:
                    loss = nn.MSELoss()(y.squeeze(), labels.squeeze())
                else:
                    loss = nn.MSELoss()(y, labels)
            elif cfg.problem == "single_label":
                loss = nn.CrossEntropyLoss()(y.view(-1, cfg.n_labels), labels.view(-1))
            elif cfg.problem == "multi_label":
                loss = nn.BCEWithLogitsLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return WithLoss(*ys)


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, bias=True, **kw)

    def prep_inputs(self, x, prev=None, use_mems=None, **kw):
        b = x.shape[0]
        dummy_token = torch.zeros((b, 1), dtype=torch.long, device=x.device)
        offset = 2
        if prev:
            x = torch.cat([x[:, -offset:], dummy_token], dim=1)
        else:
            x = torch.cat([x, dummy_token], dim=1)
        n = x.shape[1]
        pm = torch.zeros((b, n, n), dtype=torch.float, device=x.device)
        pm[:, :, -1] = 1.0
        tm = torch.zeros((b, 1, n), dtype=torch.float, device=x.device)
        tm[:, 0, -1] = 1.0
        y = {"x": x, "perm_mask": pm, "target_mapping": tm, "use_mems": use_mems}
        if prev:
            y["mems"] = tuple(x[:-offset, :, :] for x in prev)
        return y

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, y.size(-1)), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return WithLoss(*ys)


class Layer(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.het_cfg(kw)
        self.rel_attn = Attention(cfg)
        self.ffnet = FFNet(cfg.act_ffnet, cfg.drop, cfg.eps, **kw)

    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_m=None,
        **kw,
    ):
        cfg = self.cfg
        y = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_m=head_m,
        )
        output_h, output_g = y[:2]
        if output_g is not None:
            output_g = self.ffnet(output_g)
        output_h = self.ffnet(output_h)
        y = (output_h, output_g) + y[2:]
        return y


class Attention(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        cfg.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        cfg.d_model = cfg.d_model
        self.scale = 1 / (cfg.d_head**0.5)
        self.q = nn.Parameter(torch.FloatTensor(cfg.d_model, cfg.n_heads, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(cfg.d_model, cfg.n_heads, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(cfg.d_model, cfg.n_heads, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(cfg.d_model, cfg.n_heads, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(cfg.d_model, cfg.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(cfg.n_heads, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(cfg.n_heads, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(cfg.n_heads, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, cfg.n_heads, self.d_head))
        self.norm = qc.LayerNorm(cfg.d_model, cfg.eps)
        self.drop = qc.Dropout(cfg.drop)

    @staticmethod
    def rel_shift(x, klen=-1):
        s = x.shape
        y = x.reshape(s[1], s[0], s[2], s[3])
        y = y[1:, ...]
        y = y.reshape(s[0], s[1] - 1, s[2], s[3])
        # x = x[:, 0:klen, :, :]
        y = torch.index_select(y, 1, torch.arange(klen, device=y.device, dtype=torch.long))
        return y

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        s = x.shape
        y = x.reshape(s[0], s[1], s[3], s[2])
        y = y[:, :, 1:, :]
        y = y.reshape(s[0], s[1], s[2], s[3] - 1)
        y = torch.index_select(y, 3, torch.arange(klen, device=y.device, dtype=torch.long))
        # x = x[:, :, :, :klen]
        return y

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=None,
        attn_mask=None,
        head_m=None,
        **kw,
    ):
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum("ijbs,ibns->bnij", seg_mat, ef)
        y = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                y = y - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                y = y - 1e30 * torch.einsum("ijbn->bnij", attn_mask)
        y = F.softmax(y, dim=3)
        y = self.drop(y)
        if head_m is not None:
            y = y * torch.einsum("ijbn->bnij", head_m)
        y = torch.einsum("bnij,jbnd->ibnd", y, v_head_h)
        return y, torch.einsum("bnij->ijbn", y)

    def post_attention(self, h, attn_vec, residual=True):
        y = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        y = self.drop(y)
        if residual:
            y = y + h
        y = self.norm(y)
        return y

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_m=None,
        **kw,
    ):
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
            k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_m=head_m,
            )
            attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum("ibh,hnd->ibnd", g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_m=head_m,
                )
                attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_m=head_m,
                )
                attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            attn_prob = attn_prob_h, attn_prob_g
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
            k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_m=head_m,
            )
            attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        return output_h, output_g, attn_prob
