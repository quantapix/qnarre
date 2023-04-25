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

import math
import random
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.utils import logging

from .. import core as qc
from ..core import attention as qa
from ..core import forward as qf
from ..core import output as qo
from ..core import utils as qu
from ..core.embed import PosEmbed
from ..core.mlp import Classifier
from ..prep.config.bart import PreTrained


log = logging.get_logger(__name__)


class ForCausal(PreTrained):
    def __init__(self, **kw):
        kw.update(is_dec=True, is_enc_dec=False)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Decoder(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.s_vocab, bias=False, **kw)

    def forward(self, x, labels=None, **kw):
        cfg = self.cfg
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.s_vocab), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossCrosses(*ys[1:])


class ForCondGen(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        n = self.model.emb.cfg.n_embed
        self.proj = qc.Linear(cfg.d_model, n, bias=False, **kw)
        self.register_buffer("final_logits_bias", torch.zeros((1, n)))

    def forward(self, x, labels=None, x_dec_emb=None, x_dec=None, **kw):
        cfg = self.cfg
        if labels is not None:
            if x_dec is None and x_dec_emb is None:
                x_dec = qu.shift_right(labels, cfg.PAD, cfg.dec_START)
        ys = self.model(x, x_dec=x_dec, x_dec_emb=x_dec_emb, **kw)
        y = self.proj(ys[0]) + self.final_logits_bias
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.s_vocab), labels.view(-1))
        ys = (y,) + ys[1:] + (loss,)
        return qo.LossSeq2Seq(*ys)


class ForQA(PreTrained):
    def __init__(self, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, **kw)

    forward = qf.forward_seq

    def pre_proj(self, x, ys):
        y = ys[0]
        eos_m = x.eq(self.cfg.EOS)
        assert len(torch.unique_consecutive(eos_m.sum(1))) <= 1
        y = y[eos_m, :].view(y.size(0), -1, y.size(-1))
        return y[:, -1, :]


class Model(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.emb = qc.Embed(cfg.s_vocab, cfg.d_model, **kw)
        self.enc = Encoder(self.emb, **kw)
        self.dec = Decoder(self.emb, **kw)

    def forward(
        self,
        x,
        dec_head_m=None,
        dec_m=None,
        mask=None,
        x_dec_emb=None,
        x_dec=None,
        y_enc=None,
        **kw,
    ):
        cfg = self.cfg
        if x_dec is None and x_dec_emb is None:
            assert x is not None
            x_dec = qu.shift_right(x, cfg.PAD, cfg.dec_START)
        if y_enc is None:
            y_enc = self.enc(x, **kw, mask=mask)
        y = self.dec(
            x_dec, **kw, enc_m=mask, enc=y_enc[0], head_m=dec_head_m, mask=dec_m, x_emb=x_dec_emb
        )
        ys = y + y_enc
        return qo.Seq2Seq(*ys)


class Encoder(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_model
        cfg.scale = d**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, d, **kw) if tok_emb is None else tok_emb
        self.pos_emb = PosEmbed(cfg.n_pos, d, **kw)
        self.lays = qc.Stack([EncLayer(**kw) for _ in range(cfg.n_enc_lays)])
        self.norm = qc.LayerNorm(d, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.grad_checkpoint = False

    def forward(self, x, head_m=None, mask=None, x_emb=None, **kw):
        cfg = self.cfg
        if x is None:
            s = x_emb.size()[:-1]
        else:
            assert x_emb is None
            s = x.size()
            x = x.view(-1, s[-1])
        if x_emb is None:
            x_emb = self.tok_emb(x) * cfg.scale
        y = x_emb + self.pos_emb(s)
        y = self.drop(self.norm(y))
        attns = hiddens = ()
        if mask is not None:
            mask = qu.expand_mask(mask, x_emb.dtype)
        assert head_m is None or (head_m.size()[0] == (len(self.lays)))
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            if self.training and (random.uniform(0, 1) < cfg.drop_enc):
                continue
            h = head_m[i] if head_m is not None else None
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs)

                    return forward

                ys = checkpoint(create_forward(lay), y, head_m=h, mask=mask, **kw)
            else:
                ys = lay(y, head_m=h, mask=mask, **kw)
            y = ys[0]
            attns += (ys[1],)
        hiddens += (y,)
        return qo.Base(y, attns, hiddens)


class BartEncoder(PreTrained):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = qe.PosEmbed(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        mask=None,
        head_m=None,
        inputs_embeds=None,
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if mask is not None:
            mask = _expand_mask(mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_m is not None:
            if head_m.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_m should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_m.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        mask,
                        (head_m[idx] if head_m is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        mask,
                        head_m=(head_m[idx] if head_m is not None else None),
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class Decoder(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos", "eps"}, {"drop_attn": 0.0, "is_dec": False})

    def __init__(self, tok_emb=None, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_model
        cfg.scale = d**0.5 if cfg.scale else 1.0
        self.tok_emb = qc.Embed(cfg.s_vocab, d, **kw) if tok_emb is None else tok_emb
        self.pos_emb = PosEmbed(cfg.n_pos, d, **kw)
        self.lays = qc.Stack([DecLayer(**kw) for _ in range(cfg.n_dec_lays)])
        self.norm = qc.LayerNorm(d, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)
        self.grad_checkpoint = False

    def prep_dec_m(self, mask, shape, x_emb, c_len):
        y = None
        if shape[-1] > 1:
            y = qu.causal_mask(shape, x_emb.dtype, c_len=c_len).to(self.device)
        if mask is not None:
            m = qu.expand_mask(mask, x_emb.dtype, len=shape[-1])
            y = m if y is None else m + y
        return y

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
            x_emb = self.tok_emb(x) * cfg.scale
        c_len = cache[0][0].shape[2] if cache is not None else 0
        y = x_emb + self.pos_emb(s, c_len)
        y = self.drop(self.norm(y))
        attns = caches = crosses = hiddens = ()
        mask = self.prep_dec_m(mask, s, x_emb, c_len)
        if enc is not None and enc_m is not None:
            enc_m = qu.expand_mask(enc_m, x_emb.dtype, len=s[-1])
        for m in [head_m, cross_m]:
            if m is not None:
                assert m.size()[0] == (len(self.lays))
        for i, lay in enumerate(self.lays):
            hiddens += (y,)
            if self.training and (random.uniform(0, 1) < cfg.drop_dec):
                continue
            h = head_m[i] if head_m is not None else None
            c = cross_m[i] if cross_m is not None else None
            kw.update(cross_m=c, enc_m=enc_m, enc=enc, head_m=h, mask=mask)
            c = cache[i] if cache is not None else None
            if self.grad_checkpoint and self.training:

                def create_forward(x):
                    def forward(*xs):
                        return x(*xs, cache=c)

                    return forward

                ys = checkpoint(create_forward(lay), y, **kw)
            else:
                ys = lay(y, cache=c, **kw)
            y = ys[0]
            attns += (ys[1],)
            if enc is not None:
                crosses += (ys[2],)
            caches += (ys[-1],)
        hiddens += (y,)
        ys = (y, attns, caches, crosses, hiddens)
        return qo.CachesCrosses(y, attns, caches, crosses, hiddens)


class BartDecoder(PreTrained):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = qe.PosEmbed(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [BartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

    def _prepare_dec_m(self, mask, input_shape, inputs_embeds, caches_length):
        combined_mask = None
        if input_shape[-1] > 1:
            combined_mask = qu.causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                caches_length=caches_length,
            )
        if mask is not None:
            expanded_attn_mask = qu.expand_mask(
                mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_mask = (
                expanded_attn_mask if combined_mask is None else expanded_attn_mask + combined_mask
            )

        return combined_mask

    def forward(
        self,
        input_ids=None,
        mask=None,
        enc=None,
        enc_m=None,
        head_m=None,
        cross_m=None,
        caches=None,
        inputs_embeds=None,
        use_cache=None,
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and x_dec_emb at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or x_dec_emb")
        caches_length = caches[0][0].shape[2] if caches is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale
        mask = self._prepare_dec_m(mask, input_shape, inputs_embeds, caches_length)
        if enc is not None and enc_m is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            enc_m = _expand_mask(enc_m, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(input, caches_length)
        positions = positions.to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_refls = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and enc is not None) else None
        next_decoder_cache = () if use_cache else None
        for attn_mask, mask_name in zip([head_m, cross_m], ["head_m", "cross_m"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_m.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            cache = caches[idx] if caches is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for cache
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    mask,
                    enc,
                    enc_m,
                    head_m[idx] if head_m is not None else None,
                    cross_m[idx] if cross_m is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    mask=mask,
                    enc=enc,
                    enc_m=enc_m,
                    head_m=(head_m[idx] if head_m is not None else None),
                    cross_m=(cross_m[idx] if cross_m is not None else None),
                    cache=cache,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_refls += (layer_outputs[1],)

                if enc is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_refls,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            caches=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_refls,
            cross_attentions=all_cross_attentions,
        )


class EncLayer(qc.Module):
    hs = qc.Hypers(
        {"act", "d_enc_ff", "d_model", "drop_act", "n_enc_heads"},
        {"drop": 0.0, "is_dec": False},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_enc_heads, **kw)
        self.norm_refl = qc.LayerNorm(d, **kw)
        self.act = qu.activation(cfg.act)
        self.drop_act = qc.Dropout(cfg.drop_act, **kw)
        self.ff = qc.Linear(d, cfg.d_enc_ff, **kw)
        self.proj = qc.Linear(cfg.d_enc_ff, d, **kw)
        self.norm = qc.LayerNorm(d, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, **kw):
        y, a, _ = self.refl(x, **kw)
        y = self.norm_refl(x + self.drop(y))
        x = y
        y = self.drop_act(self.act(self.ff(y)))
        y = self.drop(self.proj(y))
        y = self.norm(x + y)
        if y.dtype == torch.float16 and (torch.isinf(y).any() or torch.isnan(y).any()):
            clamp = torch.finfo(y.dtype).max - 1000
            y = torch.clamp(y, min=-clamp, max=clamp)
        return y, a


class DecLayer(qc.Module):
    hs = qc.Hypers(
        {"act", "d_dec_ff", "d_model", "drop_act", "n_dec_heads"},
        {"drop": 0.0, "is_dec": False},
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_model
        self.refl = Attention(n_heads=cfg.n_dec_heads, is_dec=True, **kw)
        self.norm_refl = qc.LayerNorm(d, **kw)
        self.act = qu.activation(cfg.act)
        self.drop_act = qc.Dropout(cfg.drop_act, **kw)
        self.attn = Attention(n_heads=cfg.n_dec_heads, is_dec=True, **kw)
        self.norm_attn = qc.LayerNorm(d, **kw)
        self.ff = qc.Linear(d, cfg.d_dec_ff, **kw)
        self.proj = qc.Linear(cfg.d_dec_ff, d, **kw)
        self.norm = nn.LayerNorm(d, **kw)
        self.drop = qc.Dropout(cfg.drop, **kw)

    def forward(self, x, cache=None, cross_m=None, enc_m=None, enc=None, **kw):
        c = cache[:2] if cache is not None else None
        y, a, kv = self.refl(x, cache=c, **kw)
        y = self.norm_refl(x + self.drop(y))
        a2 = None
        if enc is not None:
            x = y
            c = cache[-2:] if cache is not None else None
            y, a2, kv2 = self.attn(y, cache=c, enc=enc, head_m=cross_m, mask=enc_m, **kw)
            y = self.norm_attn(x + self.drop(y))
            kv = kv + kv2
        x = y
        y = self.drop_act(self.act(self.ff(y)))
        y = self.proj(y)
        y = self.drop(y, p=self.dropout, training=self.training)
        return self.norm(x + y), a, a2, kv


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads"}, {"drop_attn": 0.0})

    def __init__(self, is_dec=False, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        self.is_dec = is_dec
        cfg = self.get_cfg(kw)
        d, n = cfg.d_model, cfg.n_heads
        assert d % n == 0
        cfg.s_head = s = int(d / n)
        cfg.scale = s**-0.5
        self.query = qc.Linear(d, d, **kw)
        self.key = qc.Linear(d, d, **kw)
        self.value = qc.Linear(d, d, **kw)
        self.proj = qc.Linear(d, d, **kw)
        self.drop = qc.Dropout(cfg.drop_attn, **kw)

    split_heads = qa.split_heads

    def forward(self, x, cache=None, enc=None, head_m=None, mask=None, **kw):
        cfg = self.cfg
        q = self.split_heads(self.query(x) * cfg.scale)
        if enc is None:
            k = self.split_heads(self.key(x))
            v = self.split_heads(self.value(x))
            if cache is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
        else:  # is_cross
            if cache is None:
                k = self.split_heads(self.key(enc))
                v = self.split_heads(self.value(enc))
            else:
                k, v = cache
        n, s = cfg.n_heads, cfg.s_head
        b, tgt, _ = x.size()
        h = (b * n, -1, s)
        y = torch.bmm(q.view(h), k.view(h).transpose(1, 2))
        src = k.view(h).size(1)
        assert y.size() == (b * n, tgt, src)
        if mask is not None:
            assert mask.size() == (b, 1, tgt, src)
            y = y.view(b, n, tgt, src) + mask
            y = y.view(b * n, tgt, src)
        y = F.softmax(y, dim=-1)
        if head_m is not None:
            assert head_m.size() == (n,)
            y = head_m.view(1, -1, 1, 1) * y.view(b, n, tgt, src)
            y = y.view(b * n, tgt, src)
        a = y.view(b, n, tgt, src)
        y = a.view(b * n, tgt, src)
        y = torch.bmm(self.drop(y), v.view(h))
        assert y.size() == (b * n, tgt, s)
        y = y.view(b, n, tgt, s)
        y = y.transpose(1, 2).reshape(b, tgt, cfg.d_model)
        y = self.proj(y), a, (k, v)
        return y
