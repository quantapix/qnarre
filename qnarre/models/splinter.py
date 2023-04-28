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
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embed
from ..core.mlp import Classifier, MLP, Predictor, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)


LIST = [
    "tau/splinter-base",
    "tau/splinter-base-qass",
    "tau/splinter-large",
    "tau/splinter-large-qass",
]


class SplinterEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.pos_type = getattr(config, "pos_type", "absolute")

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.pos_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class SplinterSelfAttention(qc.Module):
    def __init__(self, config, pos_type=None):
        super().__init__()
        if config.d_model % config.n_heads != 0 and not hasattr(config, "d_embed"):
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_heads})"
            )

        self.n_heads = config.n_heads
        self.attention_head_size = int(config.d_model / config.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.query = qc.Linear(config.d_model, self.all_head_size)
        self.key = qc.Linear(config.d_model, self.all_head_size)
        self.value = qc.Linear(config.d_model, self.all_head_size)

        self.drop = qc.Dropout(config.drop_attn)
        self.pos_type = pos_type or getattr(config, "pos_type", "absolute")
        if self.pos_type == "relative_key" or self.pos_type == "relative_key_query":
            self.n_pos = config.n_pos
            self.distance_embedding = qc.Embed(2 * config.n_pos - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hiddens)
        is_cross_attention = enc_hiddens is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(enc_hiddens))
            value_layer = self.transpose_for_scores(self.value(enc_hiddens))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hiddens))
            value_layer = self.transpose_for_scores(self.value(hiddens))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hiddens))
            value_layer = self.transpose_for_scores(self.value(hiddens))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.pos_type == "relative_key" or self.pos_type == "relative_key_query":
            seq_length = hiddens.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hiddens.device).view(
                -1, 1
            )
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hiddens.device).view(
                1, -1
            )
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.n_pos - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.pos_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.pos_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores + relative_position_scores_query + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.drop(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class SplinterSelfOutput(qc.Module):
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
    def __init__(self, config, pos_type=None):
        super().__init__()
        self.self = SplinterSelfAttention(config, pos_type=pos_type)
        self.output = SplinterSelfOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hiddens,
            attention_mask,
            head_mask,
            enc_hiddens,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hiddens)
        outputs = (attention_output,) + self_outputs[1:]  # add attns if we output them
        return outputs


class SplinterIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


class SplinterOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_ff, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, y, input_tensor):
        y = self.dense(y)
        y = self.drop(y)
        y = self.norm(y + input_tensor)
        return y


class Layer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = Attention(config, pos_type="absolute")
        self.intermediate = SplinterIntermediate(config)
        self.output = SplinterOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hiddens,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and enc_hiddens is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `enc_hiddens` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                enc_hiddens,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

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
        caches=None,
        y_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if y_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = caches[i] if caches is not None else None

            if self.gradient_checkpointing and self.training:
                if y_cache:
                    log.warning(
                        "`y_cache=True` is incompatible with gradient checkpointing. Setting `y_cache=False`..."
                    )
                    y_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

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
                    past_key_value,
                    output_attentions,
                )

            hiddens = layer_outputs[0]
            if y_cache:
                next_decoder_cache += (layer_outputs[-1],)
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
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return qo.CachesCrosses(
            y=hiddens,
            caches=next_decoder_cache,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
            crosses=all_cross_attentions,
        )


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = SplinterEmbeddings(config)
        self.encoder = Encoder(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        enc_hiddens=None,
        encoder_attention_mask=None,
        caches=None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = caches[0][0].shape[2] if caches is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        if self.config.is_decoder and enc_hiddens is not None:
            encoder_batch_size, encoder_sequence_length, _ = enc_hiddens.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            enc_hiddens=enc_hiddens,
            encoder_attention_mask=encoder_extended_attention_mask,
            caches=caches,
            y_cache=y_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return qo.CachesCrosses(
            y=sequence_output,
            caches=encoder_outputs.caches,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            crosses=encoder_outputs.crosses,
        )


class SplinterFullyConnectedLayer(qc.Module):
    def __init__(self, input_dim, output_dim, act="gelu"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = qc.Linear(self.input_dim, self.output_dim)
        self.act = qu.activation(act)
        self.norm = qc.LayerNorm(self.output_dim)

    def forward(self, inputs):
        hiddens = self.dense(inputs)
        hiddens = self.act(hiddens)
        hiddens = self.norm(hiddens)
        return hiddens


class QuestionAwareSpanSelectionHead(qc.Module):
    def __init__(self, config):
        super().__init__()

        self.query_start_transform = SplinterFullyConnectedLayer(config.d_model, config.d_model)
        self.query_end_transform = SplinterFullyConnectedLayer(config.d_model, config.d_model)
        self.start_transform = SplinterFullyConnectedLayer(config.d_model, config.d_model)
        self.end_transform = SplinterFullyConnectedLayer(config.d_model, config.d_model)

        self.start_classifier = qc.Linear(config.d_model, config.d_model, bias=False)
        self.end_classifier = qc.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, inputs, positions):
        _, _, dim = inputs.size()
        index = positions.unsqueeze(-1).repeat(1, 1, dim)  # [batch_size, num_positions, dim]
        gathered_reps = torch.gather(inputs, dim=1, index=index)  # [batch_size, num_positions, dim]

        query_start_reps = self.query_start_transform(
            gathered_reps
        )  # [batch_size, num_positions, dim]
        query_end_reps = self.query_end_transform(gathered_reps)  # [batch_size, num_positions, dim]
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]
        end_reps = self.end_transform(inputs)  # [batch_size, seq_length, dim]

        hiddens = self.start_classifier(query_start_reps)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # [batch_size, dim, seq_length]
        logits_beg = torch.matmul(hiddens, start_reps)

        hiddens = self.end_classifier(query_end_reps)
        end_reps = end_reps.permute(0, 2, 1)
        logits_end = torch.matmul(hiddens, end_reps)

        return logits_beg, logits_end


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        question_positions=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_positions_were_none = False
        if question_positions is None:
            if input_ids is not None:
                question_position_for_each_example = torch.argmax(
                    (torch.eq(input_ids, self.question_token_id)).int(), dim=-1
                )
            else:
                question_position_for_each_example = torch.zeros(
                    inputs_embeds.size(0),
                    dtype=torch.long,
                    layout=inputs_embeds.layout,
                    device=inputs_embeds.device,
                )
            question_positions = question_position_for_each_example.unsqueeze(-1)
            question_positions_were_none = True
        outputs = self.splinter(
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
        logits_beg, logits_end = self.splinter_qass(sequence_output, question_positions)
        if question_positions_were_none:
            logits_beg, logits_end = logits_beg.squeeze(1), logits_end.squeeze(1)
        if attention_mask is not None:
            logits_beg = logits_beg + (1 - attention_mask) * -10000.0
            logits_end = logits_end + (1 - attention_mask) * -10000.0
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = logits_beg.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(logits_beg, start_positions)
            end_loss = loss_fct(logits_end, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (logits_beg, logits_end) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        return qo.LossQA(
            loss=total_loss,
            logits_beg=logits_beg,
            logits_end=logits_end,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )
