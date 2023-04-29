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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


log = logging.get_logger(__name__)

GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neox-20b",
]


class PreTrained(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoXModel):
            module.gradient_checkpointing = value


class ForCausal(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.s_vocab, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        inputs_embeds=None,
        head_m=None,
        past_key_values=None,
        labels=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            head_m=head_m,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ForSeqClass(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        inputs_embeds=None,
        head_m=None,
        past_key_values=None,
        labels=None,
        y_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            head_m=head_m,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

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
                log.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

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
                print(pooled_logits.shape, labels.shape)
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.s_vocab, config.hidden_size)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.n_lays)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids=None,
        mask=None,
        position_ids=None,
        head_m=None,
        inputs_embeds=None,
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
        y_cache = y_cache if y_cache is not None else self.config.y_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.n_lays)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            mask = mask.view(batch_size, -1)
            mask = mask[:, None, None, :]
            mask = mask.to(dtype=self.dtype)  # fp16 compatibility
            mask = (1.0 - mask) * torch.finfo(self.dtype).min

        head_m = self.get_head_m(head_m, self.config.n_lays)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if y_cache:
                log.warning(
                    "`y_cache=True` is incompatible with gradient checkpointing. Setting `y_cache=False`..."
                )
                y_cache = False

        presents = () if y_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, cache) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for cache
                        return module(*inputs, y_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mask,
                    position_ids,
                    head_m[i],
                )
            else:
                outputs = layer(
                    hidden_states,
                    mask=mask,
                    position_ids=position_ids,
                    head_m=head_m[i],
                    cache=cache,
                    y_cache=y_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if y_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if y_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states,
        mask=None,
        position_ids=None,
        head_m=None,
        y_cache=False,
        cache=None,
        output_attentions=False,
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            mask=mask,
            position_ids=position_ids,
            cache=cache,
            head_m=head_m,
            y_cache=y_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        if y_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.n_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.n_pos
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("bias_m", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.n_pos, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(
            torch.get_default_dtype()
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_m=None,
        cache=None,
        y_cache=False,
        output_attentions=False,
    ):
        has_cache = cache is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.n_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, n_heads, 3 * head_size] --> 3 [batch, n_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_cache:
            seq_len += cache[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_cache:
            past_key = cache[0]
            past_value = cache[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if y_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, mask, head_m)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.n_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, n_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and n_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (n_heads, attn_head_size)
        # -> [bs, seq_len, n_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, n_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, n_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, n_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, n_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), n_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, mask=None, head_m=None):
        # q, k, v: [bs, n_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, n_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * n_heads, query_length, attn_head_size)
        key = key.view(batch_size * n_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * n_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(
                torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device)
                / self.norm_factor
            ),
        )
        attn_scores = attn_scores.view(batch_size, n_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_m is not None:
            attn_weights = attn_weights * head_m

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, n_pos, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = n_pos
        t = torch.arange(
            self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, n_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(
            x.device
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
