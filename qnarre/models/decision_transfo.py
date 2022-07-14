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

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.decision_transfo import PreTrained


log = logging.get_logger(__name__)


from ...pytorch_utils import Conv1D


is_amp_available = True
from torch.cuda.amp import autocast


LIST = [
    "edbeeching/decision-transformer-gym-hopper-medium",
]

# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Attention with GPT2->DecisionTransformerGPT2
class DecisionTransformerGPT2Attention(qc.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.n_pos
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.embed_dim // self.n_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.n_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by n_heads (got `embed_dim`: {self.embed_dim} and `n_heads`: {self.n_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = qc.Dropout(config.drop_attn)
        self.drop_resid = qc.Dropout(config.drop_resid)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, n_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(
            bsz * n_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device
        )

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                    -1, dk, k_seq_len
                )
                attn_weights = torch.baddbmm(
                    attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
                )
                attn_weights = attn_weights.reshape(bsz, n_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                -1, dk, k_seq_len
            )
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, n_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, n_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (n_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, n_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (n_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hiddens,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        y_cache=False,
        output_attentions=False,
    ):
        if enc_hiddens is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hiddens)
            key, value = self.c_attn(enc_hiddens).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hiddens).split(self.split_size, dim=2)

        query = self._split_heads(query, self.n_heads, self.head_dim)
        key = self._split_heads(key, self.n_heads, self.head_dim)
        value = self._split_heads(value, self.n_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if y_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.n_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.drop_resid(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attns)


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP with GPT2->DecisionTransformerGPT2
class DecisionTransformerGPT2MLP(qc.Module):
    def __init__(self, d_ff, config):
        super().__init__()
        embed_dim = config.d_model
        self.c_fc = Conv1D(d_ff, embed_dim)
        self.c_proj = Conv1D(embed_dim, d_ff)
        self.act = qu.activation(config.act)
        self.drop = qc.Dropout(config.drop_resid)

    def forward(self, x):
        y = self.c_fc(x)
        y = self.act(y)
        y = self.c_proj(y)
        y = self.drop(y)
        return y


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Block with GPT2->DecisionTransformerGPT2
class DecisionTransformerGPT2Block(qc.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        d_model = config.d_model
        inner_dim = config.n_inner if config.n_inner is not None else 4 * d_model

        self.ln_1 = qc.LayerNorm(d_model, eps=config.eps)
        self.attn = DecisionTransformerGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = qc.LayerNorm(d_model, eps=config.eps)

        if config.add_cross_attention:
            self.crossattention = DecisionTransformerGPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = qc.LayerNorm(d_model, eps=config.eps)

        self.mlp = DecisionTransformerGPT2MLP(inner_dim, config)

    def forward(
        self,
        hiddens,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
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
        attn_output = attn_outputs[0]  # output_attn: a, present, (attns)
        outputs = attn_outputs[1:]
        # residual connection
        hiddens = attn_output + residual

        if enc_hiddens is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `enc_hiddens` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hiddens
            hiddens = self.ln_cross_attn(hiddens)
            cross_attn_outputs = self.crossattention(
                hiddens,
                attention_mask=attention_mask,
                head_mask=head_mask,
                enc_hiddens=enc_hiddens,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hiddens = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attns if we output attention weights

        residual = hiddens
        hiddens = self.ln_2(hiddens)
        feed_forward_model_states = self.mlp(hiddens)
        # residual connection
        hiddens = residual + feed_forward_model_states

        if y_cache:
            outputs = (hiddens,) + outputs
        else:
            outputs = (hiddens,) + outputs[1:]

        return outputs  # hiddens, present, (attns, crosses)


class GPT2Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.d_model

        self.wte = qc.Embed(config.s_vocab, self.embed_dim)
        self.wpe = qc.Embed(config.n_pos, self.embed_dim)

        self.drop = qc.Dropout(config.drop_embed)
        self.h = nn.ModuleList(
            [DecisionTransformerGPT2Block(config, layer_idx=i) for i in range(config.n_lays)]
        )
        self.ln_f = qc.LayerNorm(self.embed_dim, eps=config.eps)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2Model.forward
    def forward(
        self,
        input_ids=None,
        caches=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
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
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        if self.config.add_cross_attention and enc_hiddens is not None:
            encoder_batch_size, encoder_sequence_length, _ = enc_hiddens.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
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
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, caches)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hiddens.device)
                # Ensure layer_past is on same device as hiddens (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hiddens.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hiddens
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hiddens.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hiddens.device)
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
                        # None for past_key_value
                        return module(*inputs, y_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hiddens,
                    None,
                    attention_mask,
                    head_mask[i],
                    enc_hiddens,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hiddens,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    enc_hiddens=enc_hiddens,
                    encoder_attention_mask=encoder_attention_mask,
                    y_cache=y_cache,
                    output_attentions=output_attentions,
                )

            hiddens = outputs[0]
            if y_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if y_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if y_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hiddens = hiddens.to("cuda:" + str(k + 1))

        hiddens = self.ln_f(hiddens)

        hiddens = hiddens.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hiddens,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return qo.CachesCrosses(
            y=hiddens,
            caches=presents,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
            crosses=all_cross_attentions,
        )


@dataclass
class Output(qo.Output):
    state_preds = None
    action_preds = None
    return_preds = None
    hiddens = None
    attns = None
    y = None


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.encoder = GPT2Model(config)
        self.embed_timestep = qc.Embed(config.max_ep_len, config.d_model)
        self.embed_return = torch.qc.Linear(1, config.d_model)
        self.embed_state = torch.qc.Linear(config.state_dim, config.d_model)
        self.embed_action = torch.qc.Linear(config.act_dim, config.d_model)
        self.embed_ln = qc.LayerNorm(config.d_model)
        self.predict_state = torch.qc.Linear(config.d_model, config.state_dim)
        self.predict_action = nn.Sequential(
            *(
                [qc.Linear(config.d_model, config.act_dim)]
                + ([nn.Tanh()] if config.action_tanh else [])
            )
        )
        self.predict_return = torch.qc.Linear(config.d_model, 1)
        self.post_init()

    def forward(
        self,
        states=None,
        actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
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

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.d_model)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.d_model).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return Output(
            y=encoder_outputs.y,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
        )
