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

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core import mlp as qm
from ..core import embed as qe
from ..core import norm as qn
from ..prep.config.llama import PreTrained

import math

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


log = logging.get_logger(__name__)


class PreTrained(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Model):
            module.gradient_checkpointing = value


class ForCausal(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        self.lm_head = nn.Linear(cfg.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            mask=mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ForSeqClass(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Model(config)
        self.score = nn.Linear(cfg.d_model, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, cfg.d_model, self.padding_idx)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = qn.RMS(cfg.d_model, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_mask
    def _prepare_decoder_mask(self, mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_mask = None
        if input_shape[-1] > 1:
            combined_mask = qu.causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                c_len=past_key_values_length,
            )
        if mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = qu.expand_mask(mask, inputs_embeds.dtype, len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_mask = (
                expanded_attn_mask if combined_mask is None else expanded_attn_mask + combined_mask
            )

        return combined_mask

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        past_key_values=None,
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
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if mask is None:
            mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        mask = self._prepare_decoder_mask(
            mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                log.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    mask=mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Layer(qc.Module):
    hs = qc.Hypers({"d_model", "add_cross", "n_inner"})

    def __init__(self, lay_i, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d = cfg.d_model
        self.attn = Attention(**kw)
        self.proj = qm.Llama(d, **kw)
        self.norm_attn = qn.RMS(d, **kw)
        self.norm = qn.RMS(d, **kw)

    def forward(self, x, mask=None, pos=None, cache=None, **kw):
        y = self.norm_attn(x)
        y, a, kv = self.attn(y, mask=mask, pos=pos, cache=cache, **kw)
        y = x + y
        x = y
        return x + self.proj(self.norm(y)), a, kv


class Attention(qc.Module):
    hs = qc.Hypers({"d_model", "n_heads", "n_pos"})

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        cfg.s_head = s = int(d / h)
        self.emb = qe.RotaryEmbed(s, **kw)
        self.query = qc.Linear(d, h * s, bias=False, **kw)
        self.key = qc.Linear(d, h * s, bias=False, **kw)
        self.value = qc.Linear(d, h * s, bias=False, **kw)
        self.proj = qc.Linear(h * s, d, bias=False, **kw)

    def _shape(self, tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, h, s).transpose(1, 2).contiguous()

    def forward(self, x, mask=None, pos=None, cache=None, **kw):
        cfg = self.cfg
        b, n_q, _ = x.size()
        d, h, s = cfg.d_model, cfg.n_heads, cfg.s_head
        q = self.query(x).view(b, n_q, h, s).transpose(1, 2)
        k = self.key(x).view(b, n_q, h, s).transpose(1, 2)
        v = self.value(x).view(b, n_q, h, s).transpose(1, 2)
        n_kv = k.shape[-2]
        if cache is not None:
            n_kv += cache[0].shape[-2]
        cos, sin = self.emb(v, seq_len=n_kv)
        q, k = qe.apply_rotary_pos_emb(q, k, cos, sin, pos)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        a = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(s)
        assert a.size() == (b, h, n_q, n_kv)
        if mask is not None:
            assert mask.size() == (b, 1, n_q, n_kv)
            a = a + mask
            a = torch.max(a, torch.tensor(torch.finfo(a.dtype).min))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(a, v)
        assert y.size() == (b, h, n_q, s)
        y = y.transpose(1, 2)
        y = y.reshape(b, n_q, d)
        return self.proj(y), a, (k, v)
