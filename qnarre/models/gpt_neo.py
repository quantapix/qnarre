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

import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Masked, Pool
from ..prep.config.gpt_neo import PreTrained


log = logging.get_logger(__name__)

LIST = [
    "EleutherAI/gpt-neo-1.3B",
]


class SelfAttention(qc.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        max_positions = config.n_pos
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        )
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.s_win))
        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = qc.Dropout(config.drop_attn)
        self.drop_resid = qc.Dropout(config.drop_resid)

        self.embed_dim = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.embed_dim // self.n_heads
        if self.head_dim * self.n_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by n_heads (got `embed_dim`: {self.embed_dim} and `n_heads`: {self.n_heads})."
            )

        self.k_proj = qc.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = qc.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = qc.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = qc.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, n_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (n_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, n_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (n_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
        self,
        hiddens,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        y_cache=False,
        output_attentions=False,
    ):
        query = self.q_proj(hiddens)
        key = self.k_proj(hiddens)
        value = self.v_proj(hiddens)
        query = self._split_heads(query, self.n_heads, self.head_dim)
        key = self._split_heads(key, self.n_heads, self.head_dim)
        value = self._split_heads(value, self.n_heads, self.head_dim)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if y_cache is True:
            present = (key, value)
        else:
            present = None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.n_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.drop_resid(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class Attention(qc.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]
        if self.attention_type in ["global", "local"]:
            self.attention = SelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def forward(
        self,
        hiddens,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        y_cache=False,
        output_attentions=False,
    ):
        return self.attention(
            hiddens,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            y_cache=y_cache,
            output_attentions=output_attentions,
        )


class MLP(qc.Module):
    def __init__(self, d_ff, config):
        super().__init__()
        embed_dim = config.d_model
        self.c_fc = qc.Linear(embed_dim, d_ff)
        self.c_proj = qc.Linear(d_ff, embed_dim)
        self.act = qu.activation(config.act)
        self.drop = qc.Dropout(config.drop_resid)

    def forward(self, x):
        y = self.c_fc(x)
        y = self.act(y)
        y = self.c_proj(y)
        y = self.drop(y)
        return y


class Block(qc.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        d_model = config.d_model
        inner_dim = config.d_ff if config.d_ff is not None else 4 * d_model
        self.ln_1 = qc.LayerNorm(d_model, eps=config.eps)
        self.attn = Attention(config, layer_id)
        self.ln_2 = qc.LayerNorm(d_model, eps=config.eps)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hiddens,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        y_cache=False,
        output_attentions=False,
    ):
        residual = hiddens
        hiddens = self.ln_1(hiddens)
        attn_outputs = self.attn(
            hiddens,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            y_cache=y_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hiddens = attn_output + residual
        residual = hiddens
        hiddens = self.ln_2(hiddens)
        feed_forward_model_states = self.mlp(hiddens)
        hiddens = residual + feed_forward_model_states
        if y_cache:
            outputs = (hiddens,) + outputs
        else:
            outputs = (hiddens,) + outputs[1:]
        return outputs


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.wte = qc.Embed(config.s_vocab, self.embed_dim)
        self.wpe = qc.Embed(config.n_pos, self.embed_dim)
        self.drop = qc.Dropout(config.drop_embed)
        self.h = nn.ModuleList([Block(config, layer_id=i) for i in range(config.n_lays)])
        self.ln_f = qc.LayerNorm(self.embed_dim, eps=config.eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        caches=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
        y_cache = y_cache if y_cache is not None else self.config.y_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if caches is None:
            past_length = 0
            caches = tuple([None] * len(self.h))
        else:
            past_length = caches[0][0].size(-2)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hiddens = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hiddens = hiddens + token_type_embeds

        hiddens = self.drop(hiddens)

        output_shape = input_shape + (hiddens.size(-1),)

        presents = () if y_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, caches)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            if self.gradient_checkpointing and self.training:
                if y_cache:
                    log.warning(
                        "`y_cache=True` is incompatible with gradient checkpointing. Setting `y_cache=False`..."
                    )
                    y_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, y_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hiddens,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hiddens,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    y_cache=y_cache,
                    output_attentions=output_attentions,
                )

            hiddens = outputs[0]
            if y_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if y_cache else 1],)

        hiddens = self.ln_f(hiddens)
        hiddens = hiddens.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)
        if not return_dict:
            return tuple(
                v
                for v in [hiddens, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return qo.BaseWithPast(
            y=hiddens,
            caches=presents,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
        )


class ForCausal(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Model(config)
        self.lm_head = qc.Linear(config.d_model, config.s_vocab, bias=False)

    def forward(
        self,
        input_ids=None,
        caches=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            caches=caches,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hiddens = transformer_outputs[0]

        lm_logits = self.lm_head(hiddens)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hiddens.dtype)
            loss = loss.to(hiddens.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            caches=transformer_outputs.caches,
            hiddens=transformer_outputs.hiddens,
            attns=transformer_outputs.attns,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq

    def post_proj(self, x):
        cfg = self.cfg
        b, _ = x.shape[:2]
        if cfg.PAD is None:
            n = -1
        else:
            assert b == 1
            n = -1 if x is None else torch.ne(x, cfg.PAD).sum(-1) - 1
        return x[torch.arange(b, device=self.device), n]
