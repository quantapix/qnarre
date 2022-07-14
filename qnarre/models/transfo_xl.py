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
# https://arxiv.org/abs/1901.02860
# https://github.com/kimiyoung/transformer-xl

import torch

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core.embed import Adaptive, Positional
from ..core.ffnet import Positionwise
from ..prep.config.transfo_xl import PreTrained

log = logging.get_logger(__name__)


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.tok_emb = Adaptive(cfg.cutoffs, div_val=cfg.div_val, **kw)
        self.pos_emb = Positional(cfg.d_model, **kw)
        if cfg.untie_r:
            q_bias = None
            r_bias = None
        else:
            q_bias = nn.Parameter(torch.FloatTensor(cfg.n_heads, cfg.d_head))
            r_bias = nn.Parameter(torch.FloatTensor(cfg.n_heads, cfg.d_head))
        self.lays = qc.Stack()
        for _ in range(cfg.n_lays):
            self.lays.append(Layer(q_bias=q_bias, r_bias=r_bias, **kw))
        self.drop = qc.Dropout(cfg.drop, **kw)

    def init_mems(self, b):
        cfg = self.cfg
        if cfg.mem_len > 0:
            p = next(self.parameters())
            kw = dict(dtype=p.dtype, device=p.device)
            return [torch.zeros(cfg.mem_len, b, cfg.d_model, **kw) for _ in range(cfg.n_lays)]
        return None

    def update_mems(self, xs, ys, mlen, qlen):
        assert len(xs) == len(ys)
        e = mlen + max(0, qlen)
        b = max(0, e - self.cfg.mem_len)
        with torch.no_grad():
            return [torch.cat([ys[i], xs[i]], dim=0)[b:e].detach() for i in range(len(xs))]

    def forward(self, x, mems=None, head_m=None, x_emb=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        if x is None:
            x_emb = x_emb.transpose(0, 1).contiguous()
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            x = x.transpose(0, 1).contiguous()
            s = x.size()
        y = self.tok_emb(x) if x_emb is None else x_emb
        n, b = s
        if mems is None:
            mems = self.init_mems(b)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + n
        pos = torch.arange(klen - 1, -1, -1.0, device=y.device, dtype=y.dtype)
        if cfg.clamp_len > 0:
            pos.clamp_(max=cfg.clamp_len)
        pos = self.drop(self.pos_emb(pos))
        ones = y.new_ones((n, klen), dtype=torch.uint8)
        if cfg.same_length:
            d = klen - cfg.mem_len
            shift = n - d if d > 0 else n
            dec_m = (torch.triu(ones, 1 + mlen) + torch.tril(ones, -shift))[:, :, None]
        else:
            dec_m = torch.triu(ones, diagonal=1 + mlen)[:, :, None]
        y = self.drop(y)
        attns = () if yo.attn else None
        hiddens = () if yo.hidden else None
        head_m = self.get_head_m2(head_m, cfg.n_lays)
        for i, lay in enumerate(self.lays):
            if yo.hidden:
                hiddens += (y,)
            m = None if mems is None else mems[i]
            ys = lay(y, pos, **kw, dec_m=dec_m, head_m=head_m[i], mems=m, yo=yo)
            y = ys[0]
            if yo.attn:
                attns += (ys[1],)
        y = self.drop(y)
        mems = None if mems is None else self.update_mems(hiddens, mems, mlen, n)
        if yo.attn:
            attns = tuple(x.permute(2, 3, 0, 1).contiguous() for x in attns)
        if yo.hidden:
            hiddens += (y,)
            hiddens = tuple(x.transpose(0, 1).contiguous() for x in hiddens)
        y = y.transpose(0, 1).contiguous()
        ys = (y, attns, hiddens, mems)
        return qo.WithMems(*ys) if yo.kw else ys


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_embed, cfg.n_labels, bias=False, **kw)

    forward = qf.forward_seq

    def post_proj(self, x):
        cfg = self.cfg
        b = (x.shape[:2] if x is not None else x_emb.shape[:2])[0]
        if cfg.PAD is None:
            n = -1
        else:
            assert b == 1
            n = -1 if x is None else torch.ne(x, cfg.PAD).sum(-1) - 1
        return x[torch.arange(b, device=self.device), n]


class LLMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        assert cfg.sample_softmax <= 0
        self.proj = Projector(
            cfg.s_vocab, cfg.d_embed, cfg.d_model, cfg.cutoffs, div_val=cfg.div_val, **kw
        )

    def tie_weights(self):
        cfg = self.cfg
        if cfg.tie_word_embeds:
            for i in range(len(self.proj.out_layers)):
                self._tie_or_clone_weights(self.proj.out_layers[i], self.model.tok_emb.lays[i])
        if cfg.tie_projs:
            for i, tie_proj in enumerate(cfg.tie_projs):
                if tie_proj and cfg.div_val == 1 and cfg.d_model != cfg.d_embed:
                    if cfg.torchscript:
                        self.proj.out_projs[i] = nn.Parameter(self.model.tok_emb.projs[0].clone())
                    else:
                        self.proj.out_projs[i] = self.model.tok_emb.projs[0]
                elif tie_proj and cfg.div_val != 1:
                    if cfg.torchscript:
                        self.proj.out_projs[i] = nn.Parameter(self.model.tok_emb.projs[i].clone())
                    else:
                        self.proj.out_projs[i] = self.model.tok_emb.projs[i]

    def init_mems(self, bsz):
        return self.model.init_mems(bsz)

    def forward(self, x, x_emb=None, labels=None, **kw):
        yo = self.get_y_opts(**kw)
        if x is None:
            assert x_emb is not None
            b, tgt = x_emb.size(0), x_emb.size(1)
        else:
            b, tgt = x.size(0), x.size(1)
        ys = self.model(x, x_emb=x_emb, **kw, yo=yo)
        xs = self.proj(ys[0][:, -tgt:], labels)
        y = xs.view(b, tgt, -1) if labels is None else ()
        loss = xs.view(b, tgt - 1) if labels is not None else None
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossMems(*ys) if yo.kw else ys


class Projector(qc.Module):
    def __init__(self, s_vocab, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super().__init__()
        self.s_vocab = s_vocab
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [s_vocab]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = qc.Stack()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(qc.Linear(d_embed, s_vocab))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))
                self.out_layers.append(qc.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, x, weight, bias, proj):
        if proj is None:
            y = F.linear(x, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            x = F.linear(x, proj.t().contiguous())
            y = F.linear(x, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias
        return y

    def forward(self, x, labels=None, keep_order=False):
        if labels is not None:
            x = x[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            x = x.view(-1, x.size(-1))
            labels = labels.view(-1)
            assert x.size(0) == labels.size(0)
        else:
            x = x.view(-1, x.size(-1))
        if self.n_clusters == 0:
            y = self._compute_logit(
                x, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0]
            )
            if labels is not None:
                y = -F.log_softmax(y, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            else:
                y = F.log_softmax(y, dim=-1)
        else:
            ws, bs = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                ws.append(weight_i)
                bs.append(bias_i)
            head_weight, head_bias, head_proj = ws[0], bs[0], self.out_projs[0]
            head_logit = self._compute_logit(x, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if labels is None:
                y = x.new_empty((head_logit.size(0), self.s_vocab))
            else:
                y = torch.zeros_like(labels, dtype=x.dtype, device=x.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                if labels is not None:
                    mask_i = (labels >= l_idx) & (labels < r_idx)
                    indices_i = mask_i.nonzero().squeeze()
                    if indices_i.numel() == 0:
                        continue
                    target_i = labels.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = x.index_select(0, indices_i)
                else:
                    hidden_i = x
                if i == 0:
                    if labels is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        y[:, : self.cutoffs[0]] = head_logprob[:, : self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = ws[i], bs[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if labels is not None:
                        logprob_i = head_logprob_i[:, cluster_prob_idx] + tail_logprob_i.gather(
                            1, target_i[:, None]
                        ).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, cluster_prob_idx, None] + tail_logprob_i
                        y[:, l_idx:r_idx] = logprob_i
                if labels is not None:
                    if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                        y.index_copy_(0, indices_i, -logprob_i)
                    else:
                        y[offset : offset + logprob_i.size(0)].copy_(-logprob_i)
                    offset += logprob_i.size(0)
        return y

    def log_prob(self, x):
        if self.n_clusters == 0:
            y = self._compute_logit(
                x, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0]
            )
            return F.log_softmax(y, dim=-1)
        else:
            ws, bs = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                ws.append(weight_i)
                bs.append(bias_i)
            head_weight, head_bias, head_proj = ws[0], bs[0], self.out_projs[0]
            head_logit = self._compute_logit(x, head_weight, head_bias, head_proj)
            y = x.new_empty((head_logit.size(0), self.s_vocab))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                beg_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    y[:, : self.cutoffs[0]] = head_logprob[:, : self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = ws[i], bs[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(x, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, -i] + tail_logprob_i
                    y[:, beg_idx, stop_idx] = logprob_i
            return y


class Layer(qc.Module):
    def __init__(self, **kw):
        super().__init__()
        self.attn = Attention(**kw)
        self.ff = Positionwise(**kw)

    def forward(self, x, r, dec_m=None, **kw):
        ys = self.attn(x, r, mask=dec_m, **kw)
        return (self.ff(ys[0]),) + ys[1:]


class Attention(qc.Module):
    hs = qc.Hypers(
        {"d_head", "d_model", "drop", "n_heads"},
        {"drop_attn": 0.0, "eps": 1e-5, "pre_norm": False},
    )

    def __init__(self, r_bias=None, q_bias=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n, h = cfg.d_model, cfg.n_heads, cfg.d_head
        cfg.scale = 1 / (h**0.5)
        self.qkv = qc.Linear(m, 3 * n * h, bias=False)
        self.r_net = qc.Linear(m, n * h, bias=False)
        if r_bias is None or q_bias is None:
            self.q_bias = nn.Parameter(torch.FloatTensor(n, h))
            self.r_bias = nn.Parameter(torch.FloatTensor(n, h))
        else:
            self.q_bias = q_bias
            self.r_bias = r_bias
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(n * h, m, bias=False, **kw)
        self.norm = qc.LayerNorm(m, **kw)

    def rel_shift(self, x, zero_triu=False):
        s = (x.size(0), 1) + x.size()[2:]
        y = torch.zeros(s, device=x.device, dtype=x.dtype)
        y = torch.cat([y, x], dim=1)
        s = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        y = y.view(*s)
        y = y[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((y.size(0), y.size(1)))
            y = y * torch.tril(ones, y.size(1) - y.size(0))[:, :, None, None]
        return y

    def forward(self, x, r, mask=None, mems=None, head_m=None, **kw):
        cfg = self.cfg
        yo = self.get_y_opts(**kw)
        y = x if mems is None else torch.cat([mems, x], 0)
        y = self.qkv(self.norm(y) if cfg.pre_norm else y)
        r = self.r_net(r)
        q, k, v = torch.chunk(a, 3, dim=-1)
        qlen, klen, rlen = x.size(0), k.size(0), r.size(0)
        q = q if mems is None else q[-qlen:]
        b, n, h = x.size(1), cfg.n_heads, cfg.d_head
        q = q.view(qlen, b, n, h)
        k = k.view(klen, b, n, h)
        v = v.view(klen, b, n, h)
        r = r.view(rlen, n, h)
        AC = torch.einsum("ibnd,jbnd->ijbn", (q + self.q_bias, k))
        BD = self.rel_shift(torch.einsum("ibnd,jnd->ijbn", (q + self.r_bias, r)))
        a = AC + BD
        a.mul_(cfg.scale)
        if mask is not None and torch.sum(mask).item():
            mask = mask == 1
            i = self.get_minus_inf()
            if mask.dim() == 2:
                a = a.float().masked_fill(mask[None, :, :, None], i).type_as(a)
            elif mask.dim() == 3:
                a = a.float().masked_fill(mask[:, :, :, None], i).type_as(a)
        a = self.drop_attn(F.softmax(a, dim=1))
        if head_m is not None:
            a = a * head_m
        y = torch.einsum("ijbn,jbnd->ibnd", (a, v))
        y = y.contiguous().view(y.size(0), y.size(1), n * h)
        y = x + self.drop(self.proj(y))
        ys = (y,) if cfg.pre_norm else (self.norm(y),)
        if yo.attn:
            ys += (a,)
        return ys
