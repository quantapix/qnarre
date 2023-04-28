# Copyright 2023 Quantapix Authors. All Rights Reserved.
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
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import attention as qa
from ..core import forward as qf
from ..core import output as qo
from ..core import utils as qu
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)


class ForChoice(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=1, **kw)

    def forward(self, x, x_emb=None, mask=None, typ=None, pos=None, labels=None, **kw):
        n = x.shape[1] if x is not None else x_emb.shape[1]
        x, mask, typ, pos = qu.view_2D(x, mask, typ, pos)
        x_emb = qu.view_3D(x_emb)
        ys = self.model(x, x_emb=x_emb, mask=mask, typ=typ, pos=pos, **kw)
        y = self.proj(ys[1]).view(-1, n)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y, labels)
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(**kw)

    forward = qf.forward_masked


class ForNext(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(n_labels=2, **kw)

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[1])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, 2), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.WithLoss(*ys)


class ForPreTraining(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Predictor(**kw)
        self.next = Classifier(n_labels=2, **kw)

    def forward(self, x, labels=None, next=None, **kw):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        n = self.next(ys[1])
        loss = None
        if labels is not None and next is not None:
            f = nn.CrossEntropyLoss()
            loss = f(y.view(-1, cfg.s_vocab), labels.view(-1)) + f(n.view(-1, 2), next.view(-1))
        ys = (y, n) + ys[2:] + (loss,)
        return qo.LossSeq(*ys)


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForSeqClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class ForTokClass(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class LMHead(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(**kw)

    def forward(self, labels=None, **kw):
        ys = self.model(**kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            y2 = y[:, :-1, :].contiguous()
            l = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(y2.view(-1, self.cfg.s_vocab), l.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCrosses(*ys)


class Masked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Predictor(**kw)

    def forward(self, x, labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            y2 = y[:, :-1, :].contiguous()
            l = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(y2.view(-1, self.cfg.s_vocab), l.view(-1))
        ys = (y,) + ys[2:] + (loss,)
        return qo.LossCrosses(*ys)


class Model(PreTrained):
    def __init__(self, add_pool=True, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.embs = Embeds(cfg.d_model, **kw)
        self.enc = Encoder(**kw)
        self.pool = Pool(**kw) if add_pool else None

    def forward(
        self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, x_emb=None, **kw
    ):
        cfg = self.cfg
        if x is None:
            s, d = x_emb.size()[:-1], x_emb.device
        else:
            assert x_emb is None
            s, d = x.size(), x.device
        c_len = cache[0][0].shape[2] if cache is not None else 0
        if mask is None:
            b, n = s
            mask = torch.ones(((b, n + c_len)), device=d)
        mask = self.get_mask(mask, s, d)
        if cfg.is_dec and enc is not None:
            if enc_m is None:
                enc_m = torch.ones(enc.size()[:2], device=d)
            enc_m = self.invert_mask(enc_m)
        else:
            enc_m = None
        head_m = self.get_head_m(head_m, cfg.n_lays)
        ys = self.embs(x, **kw, c_len=c_len, x_emb=x_emb)
        ys = self.enc(ys, **kw, cache=cache, enc_m=enc_m, enc=enc, head_m=head_m, mask=mask)
        if self.pool is not None:
            ys += (self.pool(ys[0]),)
        return qo.PoolsCrosses(*ys)


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(
        self,
        x=None,
        mask=None,
        typ=None,
        pos=None,
        head_m=None,
        x_emb=None,
        enc=None,
        enc_m=None,
        past_key_values=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            y_cache = y_cache if y_cache is not None else self.config.y_cache
        else:
            y_cache = False

        if x is not None and x_emb is not None:
            raise ValueError("You cannot specify both x and x_emb at the same time")
        elif x is not None:
            input_shape = x.size()
        elif x_emb is not None:
            input_shape = x_emb.size()[:-1]
        else:
            raise ValueError("You have to specify either x or x_emb")

        batch_size, seq_length = input_shape
        device = x.device if x is not None else x_emb.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if mask is None:
            mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if typ is None:
            if hasattr(self.embeddings, "typ"):
                buffered_typ = self.embeddings.typ[:, :seq_length]
                buffered_typ_expanded = buffered_typ.expand(batch_size, seq_length)
                typ = buffered_typ_expanded
            else:
                typ = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_mask = self.get_extended_mask(mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and enc is not None:
            encoder_batch_size, encoder_sequence_length, _ = enc.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if enc_m is None:
                enc_m = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_mask = self.invert_mask(enc_m)
        else:
            encoder_extended_mask = None

        # Prepare head mask if needed
        # 1.0 in head_m indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_m has shape [num_heads] or [n_lays x num_heads]
        # and head_m is converted to shape [n_lays x batch x num_heads x seq_length x seq_length]
        head_m = self.get_head_m(head_m, self.config.n_lays)

        embedding_output = self.embeddings(
            x=x,
            pos=pos,
            typ=typ,
            x_emb=x_emb,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            mask=extended_mask,
            head_m=head_m,
            enc=enc,
            enc_m=encoder_extended_mask,
            past_key_values=past_key_values,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class Encoder(qc.Module):
    hs = qc.Hypers({"add_cross", "n_lays"})

    def __init__(self, n_lays=None, ps={}, hs=[], **kw):
        if n_lays is not None:
            kw.update(n_lays=n_lays)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.lays = qc.Stack([Layer(**kw) for _ in range(cfg.n_lays)])
        self.grad_checkpoint = False

    def forward(self, x, cache=None, head_m=None, **kw):
        cfg = self.cfg
        y = x
        attns = caches = crosses = hiddens = ()
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            h = head_m[i] if head_m is not None else None
            c = cache[i] if cache is not None else None
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw, mask=h)
            else:
                ys = lay(y, **kw, cache=c, mask=h)
            y = ys[0]
            attns += (ys[1],)
            if cfg.add_cross:
                crosses += (ys[2],)
            caches += (ys[-1],)
        hiddens += (y,)
        return qo.CachesCrosses(y, attns, cache, crosses, hiddens)


class Layer(qc.Module):
    hs = qc.Hypers({}, {"is_dec": False})

    def __init__(self, add_cross=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        self.attn = Attention(**kw)
        if add_cross:
            assert cfg.is_dec
            self.cross = Attention(pos_type="absolute", **kw)
        self.proj = MLP(**kw)

    def forward(self, x, enc=None, cache=None, **kw):
        c = cache[:2] if cache is not None else None
        y, a, kv = self.attn(x, cache=c, **kw)
        a2 = None
        if cfg.is_dec and enc is not None:
            c = cache[-2:] if cache is not None else None
            y, a2, kv2 = self.cross(y, cache=c, enc=enc, **kw)
            kv = kv + kv2
        return self.proj(y), a, a2, kv


class Attention(qc.Module):
    hs = qc.Hypers(
        {"d_model", "drop", "n_heads", "n_pos"},
        {"drop_attn": 0.0, "pos_type": "absolute"},
    )

    def __init__(self, pos_type=None, ps={}, hs=[], **kw):
        if pos_type is not None:
            kw.update(pos_type=pos_type)
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d, n = cfg.d_model, cfg.n_heads
        assert d % n == 0  # or cfg.d_embed is not None
        cfg.s_head = s = d // n
        cfg.scale = 1 / (s**0.5)
        self.query = qc.Linear(d, d, **kw)
        self.key = qc.Linear(d, d, **kw)
        self.value = qc.Linear(d, d, **kw)
        self.drop_attn = qc.Dropout(cfg.drop_attn, **kw)
        if cfg.pos_type == "relative_key" or cfg.pos_type == "relative_key_query":
            self.emb = qc.Embed(2 * cfg.n_pos - 1, s, **kw)
        self.proj = qc.Linear(d, d, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.norm = qc.LayerNorm(d, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc_m=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
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
                k, v = cache
        a = torch.matmul(q, k.transpose(-1, -2))
        t = cfg.pos_type
        if t == "relative_key" or t == "relative_key_query":
            n_q, n_k = q.shape[2], k.shape[2]
            kw = dict(device=x.device, dtype=torch.long)
            left = torch.tensor(n_k - 1 if self.id_dec else n_q, **kw).view(-1, 1)
            right = torch.arange(n_k, **kw).view(1, -1)
            p = self.emb(left - right + cfg.n_pos - 1).to(dtype=q.dtype)
            if t == "relative_key":
                a += torch.einsum("bhld,lrd->bhlr", q, p)
            elif t == "relative_key_query":
                a += torch.einsum("bhld,lrd->bhlr", q, p) + torch.einsum("bhrd,lrd->bhlr", k, p)
        a *= cfg.scale
        if mask is not None:
            a += mask
        a = self.drop_attn(F.softmax(a, dim=-1))
        if head_m is not None:
            a *= head_m
        y = torch.matmul(a, v).permute(0, 2, 1, 3).contiguous()
        y = y.view(y.size()[:-2] + (cfg.d_model,))
        return self.norm(x + self.drop(self.proj(y))), a, (k, v)
