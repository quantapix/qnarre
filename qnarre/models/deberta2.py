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
import numpy as np
import torch
import torch.utils.checkpoint

from collections.abc import Sequence
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import forward as qf
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embed
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

log = logging.get_logger(__name__)


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Predictor(cfg.d_embed, **kw)

    forward = qf.forward_masked


class ForSeqClass(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        n_labels = getattr(config, "n_labels", 2)
        self.n_labels = n_labels

        self.deberta = Model(config)
        self.pool = Pool(config)
        output_dim = self.pool.output_dim

        self.classifier = qc.Linear(output_dim, n_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.drop if drop_out is None else drop_out
        self.drop = StableDropout(drop_out)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pool(encoder_layer)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.n_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            labeled_logits.view(-1, self.n_labels).float(), labels.view(-1)
                        )
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.n_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class ForTokClass(PreTrained):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.n_labels = config.n_labels
        self.deberta = Model(config)
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model, config.n_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.drop(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = Embed(config)
        self.encoder = Encoder(config)
        self.z_steps = 0
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hiddens = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hiddens,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return qo.Base(
            y=sequence_output,
            hiddens=encoder_outputs.hiddens if output_hidden_states else None,
            attns=encoder_outputs.attns,
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class SelfOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_model, config.d_model)
        self.norm = LayerNorm(config.d_model, config.eps)
        self.drop = StableDropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class Attention(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)
        self.config = config

    def forward(
        self,
        hiddens,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hiddens,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hiddens
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class Intermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.intermediate_act_fn(y)
        return y


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class Output(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_ff, config.d_model)
        self.norm = LayerNorm(config.d_model, config.eps)
        self.drop = StableDropout(config.drop)
        self.config = config

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->DebertaV2
class Layer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
        self,
        hiddens,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hiddens,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = qc.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.norm = LayerNorm(config.d_model, config.eps)
        self.drop = StableDropout(config.drop)
        self.config = config

    def forward(self, hiddens, residual_states, input_mask):
        out = self.conv(hiddens.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = qu.activation(self.conv_act)(self.drop(out))
        layer_norm_input = residual_states + out
        output = self.norm(layer_norm_input).to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        return output_states


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList([Layer(config) for _ in range(config.n_lays)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.n_pos

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = qc.Embed(pos_ebd_size, config.d_model)

        self.norm_rel_ebd = [
            x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.norm = LayerNorm(config.d_model, config.eps, elementwise_affine=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.norm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(
                -2
            ).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hiddens, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hiddens.size(-2)
            relative_pos = build_relative_position(
                q,
                hiddens.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def forward(
        self,
        hiddens,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hiddens, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hiddens, Sequence):
            next_kv = hiddens[0]
        else:
            next_kv = hiddens
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                output_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hiddens, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hiddens, Sequence):
                    next_kv = hiddens[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(
                v for v in [output_states, all_hidden_states, all_attentions] if v is not None
            )
        return qo.Base(
            y=output_states,
            hiddens=all_hidden_states,
            attns=all_attentions,
        )


class SelfAttention(qc.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_heads})"
            )
        self.n_heads = config.n_heads
        _attention_head_size = config.d_model // config.n_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.n_heads * self.attention_head_size
        self.query_proj = qc.Linear(config.d_model, self.all_head_size, bias=True)
        self.key_proj = qc.Linear(config.d_model, self.all_head_size, bias=True)
        self.value_proj = qc.Linear(config.d_model, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.n_pos
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.drop)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = qc.Linear(config.d_model, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = qc.Linear(config.d_model, self.all_head_size)

        self.drop = StableDropout(config.drop_attn)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hiddens,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        if query_states is None:
            query_states = hiddens
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.n_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hiddens), self.n_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hiddens), self.n_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / scale
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1, self.n_heads, attention_scores.size(-2), attention_scores.size(-1)
        )

        # bsz x height x length x dimension
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.drop(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)),
            value_layer,
        )
        context_layer = (
            context_layer.view(-1, self.n_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.n_heads
            ).repeat(query_layer.size(0) // self.n_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.n_heads
            ).repeat(query_layer.size(0) // self.n_heads, 1, 1)
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.n_heads
                ).repeat(
                    query_layer.size(0) // self.n_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.n_heads
                ).repeat(
                    query_layer.size(0) // self.n_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
                ),
            )
            score += c2p_att / scale

        # position->content
        if "p2c" in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(
                    key_layer.size(-2),
                    key_layer.size(-2),
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                ).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand(
                    [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
                ),
            ).transpose(-1, -2)
            score += p2c_att / scale

        return score


# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm
class Embed(qc.Module):
    def __init__(self, config):
        super().__init__()
        PAD = getattr(config, "PAD", 0)
        self.d_embed = getattr(config, "d_embed", config.d_model)
        self.word_embeddings = qc.Embed(config.s_vocab, self.d_embed, padding_idx=PAD)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = qc.Embed(config.n_pos, self.d_embed)

        if config.n_typ > 0:
            self.token_type_embeddings = qc.Embed(config.n_typ, self.d_embed)

        if self.d_embed != config.d_model:
            self.embed_proj = qc.Linear(self.d_embed, config.d_model, bias=False)
        self.norm = LayerNorm(config.d_model, config.eps)
        self.drop = StableDropout(config.drop)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.n_typ > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.d_embed != self.config.d_model:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.norm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.drop(embeddings)
        return embeddings


# Copied from transformers.models.deberta.modeling_deberta.Pooler
class Pool(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.drop = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hiddens):
        context_token = hiddens[:, 0]
        context_token = self.drop(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = qu.activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.d_model


# Copied from transformers.models.deberta.modeling_deberta.XSoftmax with deberta->deberta_v2
class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op(
                "Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value
            ),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
        output = softmax(g, output, dim)
        return masked_fill(
            g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8))
        )


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.drop = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, drop = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - drop)
        if drop > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(qc.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.drop = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        drop = local_context
        mask = None
    else:
        drop = local_context.drop
        drop *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if drop > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - drop)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, drop


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand(
        [query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]
    )


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand(
        [query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)]
    )


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]
