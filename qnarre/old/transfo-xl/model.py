import torch

from torch import nn
from torch.nn import functional as F

from ... import core as qc

CUDA_MAJOR = int(torch.version.cuda.split(".")[0])
CUDA_MINOR = int(torch.version.cuda.split(".")[1])


class Positional(qc.Module):
    pass


class Positionwise(qc.Module):
    pass


class Attention(qc.Module):
    hs = qc.Hypers(
        {"d_hidden", "drop", "n_heads", "d_head"},
        {"drop_attn": 0.0, "norm_eps": 1e-5, "pre_norm": False},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d, n, h = cfg.d_hidden, cfg.n_heads, cfg.d_head
        cfg.scale = 1 / (h**0.5)
        self.query = qc.Linear(d, n * h, bias=False, **kw)
        self.kv = qc.Linear(d, 2 * n * h, bias=False, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(n * h, d, bias=False, **kw)
        self.norm = qc.LayerNorm(d, **kw)

    def forward(self, x, mask=None, mems=None):
        cfg = self.cfg
        y = x if mems is None else torch.cat([mems, x], 0)
        y = self.norm(y) if cfg.pre_norm else y
        q = self.query(y)
        k, v = torch.chunk(self.kv(y), 2, -1)
        qlen, klen = x.size(0), y.size(0)
        b, n, h = x.size(1), cfg.n_heads, cfg.d_head
        q = q.view(qlen, b, n, h)
        k = k.view(klen, b, n, h)
        v = v.view(klen, b, n, h)
        a = torch.einsum("ibnd,jbnd->ijbn", (q, k))
        a.mul_(cfg.scale)
        if mask is not None and mask.any().item():
            i = self.get_minus_inf()
            if mask.dim() == 2:
                a = a.float().masked_fill(mask[None, :, :, None], i).type_as(a)
            elif mask.dim() == 3:
                a = a.float().masked_fill(mask[:, :, :, None], i).type_as(a)
        a = self.drop_attn(F.softmax(a, dim=1))
        y = torch.einsum("ijbn,jbnd->ibnd", (a, v))
        y = y.contiguous().view(y.size(0), y.size(1), n * h)
        y = x + self.drop(self.proj(y))
        return y if cfg.pre_norm else self.norm(y)


class BaseAttn(qc.Module):
    def __init__(self, tgt_len=None, ext_len=None, mem_len=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        m, n, h = cfg.d_hidden, cfg.n_heads, cfg.d_head
        cfg.scale = 1 / (h**0.5)
        self.qkv = qc.Linear(m, 3 * n * h, bias=False, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        self.proj = qc.Linear(n * h, m, bias=False, **kw)
        self.norm = qc.LayerNorm(m, **kw)

    def _parallelogram_mask(self, h, w, left=False):
        y = torch.ones((h, w)).byte()
        m = min(h, w)
        y[:m, :m] = torch.triu(y[:m, :m])
        y[-m:, -m:] = torch.tril(y[-m:, -m:])
        return y if left else y.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        kw = dict(device=x.device, dtype=x.dtype)
        if qlen > 1:
            y = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), **kw)
        else:
            y = torch.zeros(0, **kw)
        if left:
            mask = mask.flip(1)
            y = torch.cat([y, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            y = torch.cat([x, y], dim=1).expand(qlen, -1, -1, -1)
        return y.masked_select(mask[:, :, None, None]).view(qlen, klen, x.size(2), x.size(3))

    def rel_shift(self, x, zero_triu=False):
        kw = dict(device=x.device, dtype=x.dtype)
        y = torch.zeros((x.size(0), 1, *x.size()[2:]), **kw)
        y = torch.cat([y, x], dim=1)
        y = y.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = y[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x


class PartialAttn(BaseAttn):
    def __init__(self, *xs, **kw):
        super().__init__(*xs, **kw)
        cfg = self.cfg
        self.r_net = qc.Linear(cfg.d_hidden, cfg.n_heads * cfg.d_head, bias=False, **kw)

    def forward(self, x, r, q_bias, r_bias, mask=None, mems=None):
        cfg = self.cfg
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
        AC = torch.einsum("ibnd,jbnd->ijbn", (q + q_bias, k))
        BD = self.rel_shift(torch.einsum("ibnd,jnd->ijbn", (q + r_bias, r)))
        a = AC + BD
        a.mul_(cfg.scale)
        if mask is not None and mask.any().item():
            i = self.get_minus_inf()
            if mask.dim() == 2:
                a = a.float().masked_fill(mask[None, :, :, None], i).type_as(a)
            elif mask.dim() == 3:
                a = a.float().masked_fill(mask[:, :, :, None], i).type_as(a)
        a = self.drop_attn(F.softmax(a, dim=1))
        y = torch.einsum("ijbn,jbnd->ibnd", (a, v))
        y = y.contiguous().view(y.size(0), y.size(1), n * h)
        y = x + self.drop(self.proj(y))
        return y if cfg.pre_norm else self.norm(y)


class LearnableAttn(BaseAttn):
    def __init__(self, *xs, **kw):
        super().__init__(*xs, **kw)

    def forward(self, x, r, q_bias, r_bias, mask=None, mems=None):
        cfg = self.cfg
        y = x if mems is None else torch.cat([mems, x], 0)
        y = self.qkv(self.norm(y) if cfg.pre_norm else y)
        q, k, v = torch.chunk(a, 3, dim=-1)
        qlen, klen, rlen = x.size(0), k.size(0), r.size(0)
        q = q if mems is None else q[-qlen:]
        b, n, h = x.size(1), cfg.n_heads, cfg.d_head
        q = q.view(qlen, b, n, h)
        k = k.view(klen, b, n, h)
        v = v.view(klen, b, n, h)
        if klen > rlen:
            r = torch.cat([r[0:1].expand(klen - rlen, -1, -1), r], 0)
            r_bias = torch.cat([r_bias[0:1].expand(klen - r_bias.size(0), -1), r_bias], 0)
        else:
            r = r[-klen:]
            r_bias = r_bias[-klen:]
        AC = torch.einsum("ibnd,jbnd->ijbn", (q + q_bias[None], k))
        BD = self.rel_shift(torch.einsum("ibnd,jnd->ijbn", (q, r)) + r_bias[None, :, None])
        a = AC + BD
        a.mul_(cfg.scale)
        if mask is not None and mask.any().item():
            i = self.get_minus_inf()
            if mask.dim() == 2:
                a = a.float().masked_fill(mask[None, :, :, None], i).type_as(a)
            elif mask.dim() == 3:
                a = a.float().masked_fill(mask[:, :, :, None], i).type_as(a)
        a = self.drop_attn(F.softmax(a, dim=1))
        y = torch.einsum("ijbn,jbnd->ibnd", (a, v))
        y = y.contiguous().view(y.size(0), y.size(1), n * h)
        y = x + self.drop(self.proj(y))
        return y if cfg.pre_norm else self.norm(y)


class Layer(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attn = Attention(**kw)
        self.proj = Positionwise(**kw)

    def forward(self, x, dec_m=None, mems=None):
        y = self.attn(x, mask=dec_m, mems=mems)
        y = self.proj(y)
        return y


class LearnableLay(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attn = LearnableAttn(**kw)
        self.proj = Positionwise(**kw)

    def forward(self, x, r, x_bias, r_bias, dec_m=None, mems=None):
        y = self.attn(x, r, x_bias, r_bias, mask=dec_m, mems=mems)
        y = self.proj(y)
        return y


class PartialLay(qc.Module):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attn = PartialAttn(**kw)
        self.proj = Positionwise(**kw)

    def forward(self, x, r, x_bias, r_bias, dec_m=None, mems=None):
        y = self.attn(x, r, x_bias, r_bias, mask=dec_m, mems=mems)
        y = self.proj(y)
        return y


class Adaptive(qc.Module):
    pass


class MemTransformerLM(qc.Module):
    def __init__(
        self,
        tie_weight=True,
        d_embed=None,
        div_val=1,
        tie_projs=[False],
        pre_norm=False,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        cutoffs=[],
        adapt_inp=False,
        same_length=False,
        attn_type=0,
        clamp_len=-1,
        sample_softmax=-1,
    ):
        super().__init__()
        d_embed = d_hidden if d_embed is None else d_embed
        self.tok_emb = Adaptive(s_vocab, d_embed, d_hidden, cutoffs, div_val=div_val)
        self.drop = qc.Dropout(drop)
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.attn_type = attn_type
        self.lays = qc.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_lays):
                self.lays.append(
                    PartialLay(
                        d_inner,
                        tgt_len=tgt_len,
                        ext_len=ext_len,
                        mem_len=mem_len,
                        pre_norm=pre_norm,
                    )
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(cfg.n_lays):
                self.lays.append(
                    LearnableLay(
                        d_inner,
                        tgt_len=tgt_len,
                        ext_len=ext_len,
                        mem_len=mem_len,
                        pre_norm=pre_norm,
                    )
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(cfg.n_lays):
                self.lays.append(
                    Layer(
                        d_inner,
                        pre_norm=pre_norm,
                    )
                )

        self.sample_softmax = sample_softmax
        if sample_softmax > 0:
            self.out_layer = qc.Linear(d_hidden, s_vocab)
            if tie_weight:
                self.out_layer.weight = self.tok_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(s_vocab, sample_softmax)
        else:
            self.crit = ProjecLogSoftmax(s_vocab, d_embed, d_hidden, cutoffs, div_val=div_val)
            if tie_weight:
                for i in range(len(self.crit.lays)):
                    self.crit.lays[i].weight = self.tok_emb.lays[i].weight
            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_hidden != d_embed:
                        self.crit.projs[i] = self.tok_emb.projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.projs[i] = self.tok_emb.projs[i]
        self.same_length = same_length
        self.clamp_len = clamp_len
        self._create_params()

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        assert len(hids) == len(mems), "len(hids) != len(mems)"
        with torch.no_grad():
            ys = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                ys.append(cat[beg_idx:end_idx].detach())
        return ys

    def _forward(self, x, mems=None):
        qlen, bsz = x.size()
        word_emb = self.tok_emb(x)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_m = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[
                :, :, None
            ]
        else:
            dec_m = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            y = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)
            hids.append(y)
            for i, lay in enumerate(self.lays):
                mems_i = None if mems is None else mems[i]
                y = lay(
                    y,
                    pos_emb,
                    self.r_w_bias,
                    self.r_r_bias,
                    dec_m=dec_m,
                    mems=mems_i,
                )
                hids.append(y)
        elif self.attn_type == 1:  # learnable
            y = self.drop(word_emb)
            hids.append(y)
            for i, lay in enumerate(self.lays):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]
                mems_i = None if mems is None else mems[i]
                y = lay(
                    y,
                    r_emb,
                    self.r_w_bias[i],
                    r_bias,
                    dec_m=dec_m,
                    mems=mems_i,
                )
                hids.append(y)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            y = self.drop(word_emb + pos_emb[-qlen:])
            hids.append(y)
            for i, lay in enumerate(self.lays):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                y = lay(y, dec_m=dec_m, mems=mems_i)
                hids.append(y)
        elif self.attn_type == 3:
            y = self.drop(word_emb)
            hids.append(y)
            for i, lay in enumerate(self.lays):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                y += self.r_emb[i][-qlen:].view(qlen, 1, -1)
                y = lay(y, dec_m=dec_m, mems=mems_i)
                hids.append(y)
        y = self.drop(y)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return y, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.tok_emb, self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)
        return [loss] if new_mems is None else [loss] + new_mems


class AdaptLogSoftmax(qc.Module):
    def __init__(self, in_features, n_classes, cutoffs, keep_order=False):
        super().__init__()
        cutoffs = list(cutoffs)
        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) >= (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any([int(c) != c for c in cutoffs])
        ):

            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.in_features))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.keep_order = keep_order

    def forward(self, hidden, target, weight, bias, keep_order=False):
        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        head_weight = torch.cat([weight[: self.shortlist_size], self.cluster_weight], dim=0)
        head_bias = torch.cat([bias[: self.shortlist_size], self.cluster_bias], dim=0)

        head_logit = F.linear(hidden, head_weight, bias=head_bias)
        head_logprob = F.log_softmax(head_logit, dim=1)

        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, h_idx = cutoff_values[i], cutoff_values[i + 1]

            mask_i = (target >= l_idx) & (target < h_idx)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                weight_i = weight[l_idx:h_idx]
                bias_i = bias[l_idx:h_idx]

                hidden_i = hidden.index_select(0, indices_i)

                tail_logit_i = F.linear(hidden_i, weight_i, bias=bias_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                    1, target_i[:, None]
                ).squeeze(1)

            if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

            offset += logprob_i.size(0)

        return nll


class ProjecLogSoftmax(qc.Module):
    def __init__(self, s_vocab, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjecLogSoftmax, self).__init__()
        self.cutoffs = cutoffs + [s_vocab]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.lays = qc.ModuleList()
        self.projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.projs.append(None)
            self.lays.append(qc.Linear(d_embed, s_vocab))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
                self.lays.append(qc.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False):
        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden, self.lays[0].weight, self.lays[0].bias, self.projs[0]
            )
            nll = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.lays[0].weight[l_idx:r_idx]
                    bias_i = self.lays[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.lays[i].weight
                    bias_i = self.lays[i].bias

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                        1, target_i[:, None]
                    ).squeeze(1)

                if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels):
        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding, bias, labels, inputs, sampler):
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = torch.einsum("ijk,ijk->ij", [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum("lk,ijk->ijl", [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")

    parser.add_argument("--n_lays", type=int, default=4, help="")
    parser.add_argument("--n_rel_layer", type=int, default=4, help="")
    parser.add_argument("--n_heads", type=int, default=2, help="")
    parser.add_argument("--d_head", type=int, default=2, help="")
    parser.add_argument("--d_hidden", type=int, default=200, help="")
    parser.add_argument("--d_embed", type=int, default=200, help="")
    parser.add_argument("--d_inner", type=int, default=200, help="")
    parser.add_argument("--drop", type=float, default=0.0, help="")
    parser.add_argument("--cuda", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=1111, help="")
    parser.add_argument("--multi_gpu", action="store_true", help="")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.s_vocab = 10000

    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, args.s_vocab).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.s_vocab // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(
                tie_weight=True,
                d_embed=d_embed,
                div_val=div_val,
                tie_projs=tie_projs,
                pre_norm=True,
                tgt_len=tgt_len,
                ext_len=ext_len,
                mem_len=mem_len,
                cutoffs=cutoffs,
                attn_type=0,
            ).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print("batch {}".format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
