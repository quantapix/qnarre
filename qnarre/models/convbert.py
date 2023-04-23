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

import math
import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, FFNet, Masker, Pool
from ..prep.config.convbert import PreTrained


from torch.nn import CrossEntropyLoss

from ...modeling_utils import SequenceSummary
from ...pytorch_utils import (
    apply_chunking_to_forward,
)

from . import bert

log = logging.get_logger(__name__)

LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
]


class ConvBertEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_embed, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_embed)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_embed)
        self.norm = qc.LayerNorm(config.d_embed, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class SeparableConv1D(qc.Module):
    def __init__(self, config, input_filters, output_filters, kernel_size, **kw):
        super().__init__()
        self.depthwise = qc.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pointwise = qc.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        self.depthwise.weight.data.normal_(mean=0.0, std=config.init_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.init_range)

    def forward(self, hiddens):
        x = self.depthwise(hiddens)
        x = self.pointwise(x)
        x += self.bias
        return x


class ConvBertSelfAttention(qc.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads != 0 and not hasattr(config, "d_embed"):
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_heads})"
            )

        new_num_attention_heads = config.n_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.n_heads
            self.n_heads = 1
        else:
            self.n_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        self.conv_kernel_size = config.conv_kernel_size
        if config.d_model % self.n_heads != 0:
            raise ValueError("d_model should be divisible by n_heads")

        self.attention_head_size = config.d_model // config.n_heads
        self.all_head_size = self.n_heads * self.attention_head_size

        self.query = qc.Linear(config.d_model, self.all_head_size)
        self.key = qc.Linear(config.d_model, self.all_head_size)
        self.value = qc.Linear(config.d_model, self.all_head_size)

        self.key_conv_attn_layer = SeparableConv1D(
            config, config.d_model, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = qc.Linear(self.all_head_size, self.n_heads * self.conv_kernel_size)
        self.conv_out_layer = qc.Linear(config.d_model, self.all_head_size)

        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1],
            padding=[int((self.conv_kernel_size - 1) / 2), 0],
        )

        self.drop = qc.Dropout(config.drop_attn)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hiddens)
        batch_size = hiddens.size(0)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if enc_hiddens is not None:
            mixed_key_layer = self.key(enc_hiddens)
            mixed_value_layer = self.value(enc_hiddens)
        else:
            mixed_key_layer = self.key(hiddens)
            mixed_value_layer = self.value(hiddens)

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hiddens.transpose(1, 2))
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)

        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

        conv_out_layer = self.conv_out_layer(hiddens)
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        conv_out_layer = F.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        conv_out_layer = torch.reshape(
            conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size]
        )
        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.drop(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        conv_out = torch.reshape(
            conv_out_layer, [batch_size, -1, self.n_heads, self.attention_head_size]
        )
        context_layer = torch.cat([context_layer, conv_out], 2)

        new_context_layer_shape = context_layer.size()[:-2] + (
            self.head_ratio * self.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ConvBertSelfOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_model, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


class Attention(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ConvBertSelfAttention(config)
        self.output = ConvBertSelfOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hiddens,
            attention_mask,
            head_mask,
            enc_hiddens,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hiddens)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class GroupedLinearLayer(qc.Module):
    def __init__(self, input_size, output_size, n_groups):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_groups = n_groups
        self.group_in_dim = self.input_size // self.n_groups
        self.group_out_dim = self.output_size // self.n_groups
        self.weight = nn.Parameter(
            torch.empty(self.n_groups, self.group_in_dim, self.group_out_dim)
        )
        self.bias = nn.Parameter(torch.empty(output_size))

    def forward(self, hiddens):
        batch_size = list(hiddens.size())[0]
        x = torch.reshape(hiddens, [-1, self.n_groups, self.group_in_dim])
        x = x.permute(1, 0, 2)
        x = torch.matmul(x, self.weight)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias
        return x


class ConvBertIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.n_groups == 1:
            self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        else:
            self.dense = GroupedLinearLayer(
                input_size=cfg.d_model,
                output_size=cfg.d_ff,
                n_groups=cfg.n_groups,
            )
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


class ConvBertOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_groups == 1:
            self.dense = qc.Linear(config.d_ff, config.d_model)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.d_ff,
                output_size=config.d_model,
                n_groups=config.n_groups,
            )
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


class Layer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder
            self.crossattention = Attention(config)
        self.intermediate = ConvBertIntermediate(config)
        self.output = ConvBertOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hiddens,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and enc_hiddens is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `enc_hiddens` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )
            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_attention_mask,
                head_mask,
                enc_hiddens,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.n_lays)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hiddens,
                    attention_mask,
                    layer_head_mask,
                    enc_hiddens,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hiddens,
                    attention_mask,
                    layer_head_mask,
                    enc_hiddens,
                    encoder_attention_mask,
                    output_attentions,
                )
            hiddens = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hiddens,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return qo.BaseWithCrossAttentions(
            y=hiddens,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
            crosses=all_cross_attentions,
        )


class ConvBertPredictionHeadTransform(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_model)
        self.act = qu.activation(cfg.act)
        self.norm = qc.LayerNorm(cfg.d_model, eps=cfg.eps)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        y = self.norm(y)
        return y


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ConvBertEmbeddings(config)
        if config.d_embed != config.d_model:
            self.embeddings_project = qc.Linear(config.d_embed, config.d_model)
        self.encoder = Encoder(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
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
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)

        hiddens = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if hasattr(self, "embeddings_project"):
            hiddens = self.embeddings_project(hiddens)

        hiddens = self.encoder(
            hiddens,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hiddens


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = qc.Linear(config.d_model, 1)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        pooled_output = self.sequence_summary(sequence_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return qo.WithLoss(
            loss=loss,
            logits=reshaped_logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(cfg.d_embed, **kw)

    forward = qf.forward_masked


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(cfg.d_model, **kw)

    forward = qf.forward_seq


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
