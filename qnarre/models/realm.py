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

from dataclasses import dataclass
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


from ...pytorch_utils import (
    apply_chunking_to_forward,
)

log = logging.get_logger(__name__)

LIST = [
    "google/realm-cc-news-pretrained-embedder",
    "google/realm-cc-news-pretrained-encoder",
    "google/realm-cc-news-pretrained-scorer",
    "google/realm-cc-news-pretrained-openqa",
    "google/realm-orqa-nq-openqa",
    "google/realm-orqa-nq-reader",
    "google/realm-orqa-wq-openqa",
    "google/realm-orqa-wq-reader",
]


# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->Realm
class RealmEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.pos_type = getattr(config, "pos_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

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
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.pos_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Realm
class RealmSelfAttention(qc.Module):
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
            # reuse k,v, crosses
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
            # Apply the attention mask is (precomputed for all layers in RealmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.drop(attention_probs)

        # Mask heads if we want to
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


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Realm
class RealmSelfOutput(qc.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Realm
class Attention(qc.Module):
    def __init__(self, config, pos_type=None):
        super().__init__()
        self.self = RealmSelfAttention(config, pos_type=pos_type)
        self.output = RealmSelfOutput(config)

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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Realm
class RealmIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Realm
class RealmOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = qc.Linear(config.d_ff, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        hiddens = self.norm(hiddens + input_tensor)
        return hiddens


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Realm
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
        self.intermediate = RealmIntermediate(config)
        self.output = RealmOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hiddens,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attns if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and enc_hiddens is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `enc_hiddens` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
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
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attns if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Realm
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


@dataclass
class RealmEmbedderOutput(ModelOutput):
    projected_score = None
    hiddens = None
    attns = None


@dataclass
class RealmScorerOutput(ModelOutput):
    relevance_score = None
    query_score = None
    candidate_score = None


@dataclass
class RealmReaderOutput(ModelOutput):
    loss = None
    retriever_loss = None
    reader_loss = None
    retriever_correct = None
    reader_correct = None
    block_idx = None
    candidate = None
    start_pos = None
    end_pos = None
    hiddens = None
    attns = None


@dataclass
class RealmForOpenQAOutput(ModelOutput):
    reader_output = None
    predicted_answer_ids = None


class RealmPredictionHeadTransform(qc.Module):
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


class RealmLMPredictionHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RealmPredictionHeadTransform(config)
        self.decoder = qc.Linear(config.d_model, config.s_vocab, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.s_vocab))
        self.decoder.bias = self.bias

    def forward(self, x):
        y = self.transform(x)
        y = self.decoder(y)
        return y


class RealmOnlyMLMHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RealmScorerProjection(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)
        self.dense = qc.Linear(config.d_model, config.retriever_proj_size)
        self.norm = qc.LayerNorm(config.retriever_proj_size, eps=config.eps)

    def forward(self, hiddens):
        hiddens = self.dense(hiddens)
        hiddens = self.norm(hiddens)
        return hiddens


class RealmReaderProjection(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense_intermediate = qc.Linear(config.d_model, config.span_hidden_size * 2)
        self.dense_output = qc.Linear(config.span_hidden_size, 1)
        self.layer_normalization = qc.LayerNorm(
            config.span_hidden_size, eps=config.reader_layer_norm_eps
        )
        self.relu = nn.ReLU()

    def forward(self, hiddens, block_mask):
        def span_candidates(masks):
            _, max_sequence_len = masks.shape

            def _spans_given_width(width):
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return current_starts, current_ends

            starts, ends = zip(
                *(_spans_given_width(w + 1) for w in range(self.config.max_span_width))
            )

            # [num_spans]
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)

            # [num_retrievals, num_spans]
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks

            return starts, ends, span_masks

        def mask_to_score(mask):
            return (1.0 - mask.type(torch.float32)) * -10000.0

        # [reader_beam_size, max_sequence_len, span_hidden_size * 2]
        hiddens = self.dense_intermediate(hiddens)
        # [reader_beam_size, max_sequence_len, span_hidden_size]
        start_projection, end_projection = hiddens.chunk(2, dim=-1)

        candidate_starts, candidate_ends, candidate_mask = span_candidates(block_mask)

        candidate_start_projections = torch.index_select(
            start_projection, dim=1, index=candidate_starts
        )
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections

        # [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.relu(candidate_hidden)
        # [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.layer_normalization(candidate_hidden)
        # [reader_beam_size, num_candidates]
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
        # [reader_beam_size, num_candidates]
        reader_logits += mask_to_score(candidate_mask)

        return reader_logits, candidate_starts, candidate_ends


class Model(PreTrained):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RealmEmbeddings(config)
        self.encoder = Encoder(config)

        self.pool = Pool(config) if add_pooling_layer else None

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
        pooled_output = self.pool(sequence_output) if self.pool is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return qo.BaseWithPoolingAndCrossAttentions(
            y=sequence_output,
            pools=pooled_output,
            caches=encoder_outputs.caches,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            crosses=encoder_outputs.crosses,
        )


class RealmEmbedder(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.realm = Model(self.config)
        self.cls = RealmScorerProjection(self.config)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        realm_outputs = self.realm(
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

        # [batch_size, d_model]
        pools = realm_outputs[1]
        # [batch_size, retriever_proj_size]
        projected_score = self.cls(pools)

        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(
                projected_score=projected_score,
                hiddens=realm_outputs.hiddens,
                attns=realm_outputs.attns,
            )


class RealmScorer(PreTrained):
    def __init__(self, config, query_embedder=None):
        super().__init__(config)

        self.embedder = RealmEmbedder(self.config)

        self.query_embedder = query_embedder if query_embedder is not None else self.embedder

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        candidate_input_ids=None,
        candidate_attention_mask=None,
        candidate_token_type_ids=None,
        candidate_inputs_embeds=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or input_embeds.")

        if candidate_input_ids is None and candidate_inputs_embeds is None:
            raise ValueError(
                "You have to specify either candidate_input_ids or candidate_inputs_embeds."
            )

        query_outputs = self.query_embedder(
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

        # [batch_size * num_candidates, candidate_seq_len]
        (
            flattened_input_ids,
            flattened_attention_mask,
            flattened_token_type_ids,
        ) = self._flatten_inputs(
            candidate_input_ids, candidate_attention_mask, candidate_token_type_ids
        )

        candidate_outputs = self.embedder(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=candidate_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size, retriever_proj_size]
        query_score = query_outputs[0]
        # [batch_size * num_candidates, retriever_proj_size]
        candidate_score = candidate_outputs[0]
        # [batch_size, num_candidates, retriever_proj_size]
        candidate_score = candidate_score.view(
            -1, self.config.num_candidates, self.config.retriever_proj_size
        )
        # [batch_size, num_candidates]
        relevance_score = torch.einsum("BD,BND->BN", query_score, candidate_score)

        if not return_dict:
            return relevance_score, query_score, candidate_score

        return RealmScorerOutput(
            relevance_score=relevance_score,
            query_score=query_score,
            candidate_score=candidate_score,
        )


class RealmKnowledgeAugEncoder(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.realm = Model(self.config)
        self.cls = RealmOnlyMLMHead(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        relevance_score=None,
        labels=None,
        mlm_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            flattened_input_ids,
            flattened_attention_mask,
            flattened_token_type_ids,
        ) = self._flatten_inputs(input_ids, attention_mask, token_type_ids)

        joint_outputs = self.realm(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size * num_candidates, joint_seq_len, d_model]
        joint_output = joint_outputs[0]
        # [batch_size * num_candidates, joint_seq_len, s_vocab]
        prediction_scores = self.cls(joint_output)
        # [batch_size, num_candidates]
        candidate_score = relevance_score

        masked_lm_loss = None
        if labels is not None:
            if candidate_score is None:
                raise ValueError(
                    "You have to specify `relevance_score` when `labels` is specified in order to compute loss."
                )

            batch_size, seq_length = labels.size()

            if mlm_mask is None:
                mlm_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                mlm_mask = mlm_mask.type(torch.float32)

            # Compute marginal log-likelihood
            loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token

            # [batch_size * num_candidates * joint_seq_len, s_vocab]
            mlm_logits = prediction_scores.view(-1, self.config.s_vocab)
            # [batch_size * num_candidates * joint_seq_len]
            mlm_targets = labels.tile(1, self.config.num_candidates).view(-1)
            # [batch_size, num_candidates, joint_seq_len]
            masked_lm_log_prob = -loss_fct(mlm_logits, mlm_targets).view(
                batch_size, self.config.num_candidates, seq_length
            )
            # [batch_size, num_candidates, 1]
            candidate_log_prob = candidate_score.log_softmax(-1).unsqueeze(-1)
            # [batch_size, num_candidates, joint_seq_len]
            joint_gold_log_prob = candidate_log_prob + masked_lm_log_prob
            # [batch_size, joint_seq_len]
            marginal_gold_log_probs = joint_gold_log_prob.logsumexp(1)
            # []
            masked_lm_loss = -torch.nansum(
                torch.sum(marginal_gold_log_probs * mlm_mask) / torch.sum(mlm_mask)
            )

        if not return_dict:
            output = (prediction_scores,) + joint_outputs[2:4]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hiddens=joint_outputs.hiddens,
            attns=joint_outputs.attns,
        )


class RealmReader(PreTrained):
    _keys_to_ignore_on_load_unexpected = [r"pooler", "cls"]

    def __init__(self, config):
        super().__init__(config)
        self.n_labels = config.n_labels

        self.realm = Model(config)
        self.cls = RealmOnlyMLMHead(config)
        self.qa_outputs = RealmReaderProjection(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        relevance_score=None,
        block_mask=None,
        start_positions=None,
        end_positions=None,
        has_answers=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if relevance_score is None:
            raise ValueError("You have to specify `relevance_score` to calculate logits and loss.")
        if block_mask is None:
            raise ValueError(
                "You have to specify `block_mask` to separate question block and evidence block."
            )
        if token_type_ids.size(1) < self.config.max_span_width:
            raise ValueError(
                "The input sequence length must be greater than or equal to config.max_span_width."
            )
        outputs = self.realm(
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

        # [reader_beam_size, joint_seq_len, d_model]
        sequence_output = outputs[0]

        # [reader_beam_size, num_candidates], [num_candidates], [num_candidates]
        reader_logits, candidate_starts, candidate_ends = self.qa_outputs(
            sequence_output, block_mask[0 : self.config.reader_beam_size]
        )
        # [searcher_beam_size, 1]
        retriever_logits = torch.unsqueeze(relevance_score[0 : self.config.reader_beam_size], -1)
        # [reader_beam_size, num_candidates]
        reader_logits += retriever_logits
        # []
        predicted_block_index = torch.argmax(torch.max(reader_logits, dim=1).values)
        # []
        predicted_candidate = torch.argmax(torch.max(reader_logits, dim=0).values)
        # [1]
        predicted_start = torch.index_select(candidate_starts, dim=0, index=predicted_candidate)
        # [1]
        predicted_end = torch.index_select(candidate_ends, dim=0, index=predicted_candidate)

        total_loss = None
        retriever_loss = None
        reader_loss = None
        retriever_correct = None
        reader_correct = None
        if start_positions is not None and end_positions is not None and has_answers is not None:

            def compute_correct_candidates(
                candidate_starts, candidate_ends, gold_starts, gold_ends
            ):
                """Compute correct span."""
                # [reader_beam_size, num_answers, num_candidates]
                is_gold_start = torch.eq(
                    torch.unsqueeze(torch.unsqueeze(candidate_starts, 0), 0),
                    torch.unsqueeze(gold_starts, -1),
                )
                is_gold_end = torch.eq(
                    torch.unsqueeze(torch.unsqueeze(candidate_ends, 0), 0),
                    torch.unsqueeze(gold_ends, -1),
                )

                # [reader_beam_size, num_candidates]
                return torch.any(torch.logical_and(is_gold_start, is_gold_end), 1)

            def marginal_log_loss(logits, is_correct):
                """Loss based on the negative marginal log-likelihood."""

                def mask_to_score(mask):
                    return (1.0 - mask.type(torch.float32)) * -10000.0

                # []
                log_numerator = torch.logsumexp(logits + mask_to_score(is_correct), dim=-1)
                log_denominator = torch.logsumexp(logits, dim=-1)
                return log_denominator - log_numerator

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # `-1` is reserved for no answer.
            ignored_index = sequence_output.size(1)
            start_positions = start_positions.clamp(-1, ignored_index)
            end_positions = end_positions.clamp(-1, ignored_index)

            retriever_correct = has_answers
            any_retriever_correct = torch.any(retriever_correct)

            reader_correct = compute_correct_candidates(
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
                gold_starts=start_positions[0 : self.config.reader_beam_size],
                gold_ends=end_positions[0 : self.config.reader_beam_size],
            )
            any_reader_correct = torch.any(reader_correct)

            retriever_loss = marginal_log_loss(relevance_score, retriever_correct)
            reader_loss = marginal_log_loss(reader_logits.view(-1), reader_correct.view(-1))
            retriever_loss *= any_retriever_correct.type(torch.float32)
            reader_loss *= any_reader_correct.type(torch.float32)

            total_loss = (retriever_loss + reader_loss).mean()

        if not return_dict:
            output = (
                predicted_block_index,
                predicted_candidate,
                predicted_start,
                predicted_end,
            ) + outputs[2:]
            return (
                (
                    (total_loss, retriever_loss, reader_loss, retriever_correct, reader_correct)
                    + output
                )
                if total_loss is not None
                else output
            )

        return RealmReaderOutput(
            loss=total_loss,
            retriever_loss=retriever_loss,
            reader_loss=reader_loss,
            retriever_correct=retriever_correct,
            reader_correct=reader_correct,
            block_idx=predicted_block_index,
            candidate=predicted_candidate,
            start_pos=predicted_start,
            end_pos=predicted_end,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class ForQA(PreTrained):
    def __init__(self, config, retriever=None):
        super().__init__(config)
        self.embedder = RealmEmbedder(config)
        self.reader = RealmReader(config)
        self.register_buffer(
            "block_emb",
            torch.zeros(()).new_empty(
                size=(config.num_block_records, config.retriever_proj_size),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )
        self.retriever = retriever

    @property
    def searcher_beam_size(self):
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    def block_embedding_to(self, device):
        self.block_emb = self.block_emb.to(device)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        answer_ids=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError("The batch_size of the inputs must be 1.")

        question_outputs = self.embedder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        question_projection = question_outputs[0]
        batch_scores = torch.einsum(
            "BD,QD->QB", self.block_emb, question_projection.to(self.block_emb.device)
        )
        _, retrieved_block_ids = torch.topk(batch_scores, k=self.searcher_beam_size, dim=-1)
        retrieved_block_ids = retrieved_block_ids.squeeze()
        retrieved_block_emb = torch.index_select(self.block_emb, dim=0, index=retrieved_block_ids)
        has_answers, start_pos, end_pos, concat_inputs = self.retriever(
            retrieved_block_ids.cpu(), input_ids, answer_ids, max_length=self.config.reader_seq_len
        )

        concat_inputs = concat_inputs.to(self.reader.device)
        block_mask = concat_inputs.special_tokens_mask.type(torch.bool).to(
            device=self.reader.device
        )
        block_mask.logical_not_().logical_and_(concat_inputs.token_type_ids.type(torch.bool))

        if has_answers is not None:
            has_answers = torch.tensor(has_answers, dtype=torch.bool, device=self.reader.device)
            start_pos = torch.tensor(start_pos, dtype=torch.long, device=self.reader.device)
            end_pos = torch.tensor(end_pos, dtype=torch.long, device=self.reader.device)
        retrieved_logits = torch.einsum(
            "D,BD->B", question_projection.squeeze(), retrieved_block_emb.to(self.reader.device)
        )
        reader_output = self.reader(
            input_ids=concat_inputs.input_ids[0 : self.config.reader_beam_size],
            attention_mask=concat_inputs.attention_mask[0 : self.config.reader_beam_size],
            token_type_ids=concat_inputs.token_type_ids[0 : self.config.reader_beam_size],
            relevance_score=retrieved_logits,
            block_mask=block_mask,
            has_answers=has_answers,
            start_positions=start_pos,
            end_positions=end_pos,
            return_dict=True,
        )
        predicted_block = concat_inputs.input_ids[reader_output.block_idx]
        predicted_answer_ids = predicted_block[reader_output.start_pos : reader_output.end_pos + 1]
        if not return_dict:
            return reader_output, predicted_answer_ids

        return RealmForOpenQAOutput(
            reader_output=reader_output,
            predicted_answer_ids=predicted_answer_ids,
        )
