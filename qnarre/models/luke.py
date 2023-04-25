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
import math

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Masked, Pool
from ..prep.config.bert import PreTrained


log = logging.get_logger(__name__)

from ...pytorch_utils import apply_chunking_to_forward
from ...utils import ModelOutput

LIST = [
    "studio-ousia/luke-base",
    "studio-ousia/luke-large",
]


@dataclass
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    entity_last_hidden_state = None
    entity_hidden_states = None


@dataclass
class BaseLukeModelOutput(BaseModelOutput):
    entity_last_hidden_state = None
    entity_hidden_states = None


@dataclass
class EntityClassificationOutput(ModelOutput):
    loss = None
    logits = None
    hiddens = None
    entity_hidden_states = None
    attns = None


@dataclass
class EntityPairClassificationOutput(ModelOutput):
    loss = None
    logits = None
    hiddens = None
    entity_hidden_states = None
    attns = None


@dataclass
class EntitySpanClassificationOutput(ModelOutput):
    loss = None
    logits = None
    hiddens = None
    entity_hidden_states = None
    attns = None


class LukeEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        self.padding_idx = config.PAD
        self.position_embeddings = qc.Embed(
            config.n_pos, config.d_model, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
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

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LukeEntityEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.entity_embeddings = qc.Embed(
            config.entity_vocab_size, config.entity_emb_size, padding_idx=0
        )
        if config.entity_emb_size != config.d_model:
            self.entity_embedding_dense = qc.Linear(
                config.entity_emb_size, config.d_model, bias=False
            )

        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)

        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(
        self,
        entity_ids,
        position_ids,
        token_type_ids=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.d_model:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(
            min=1e-7
        )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)

        return embeddings


class LukeSelfAttention(qc.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads != 0 and not hasattr(config, "d_embed"):
            raise ValueError(
                f"The hidden size {config.d_model,} is not a multiple of the number of attention "
                f"heads {config.n_heads}."
            )

        self.n_heads = config.n_heads
        self.attention_head_size = int(config.d_model / config.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size
        self.use_entity_aware_attention = config.use_entity_aware_attention

        self.query = qc.Linear(config.d_model, self.all_head_size)
        self.key = qc.Linear(config.d_model, self.all_head_size)
        self.value = qc.Linear(config.d_model, self.all_head_size)

        if self.use_entity_aware_attention:
            self.w2e_query = qc.Linear(config.d_model, self.all_head_size)
            self.e2w_query = qc.Linear(config.d_model, self.all_head_size)
            self.e2e_query = qc.Linear(config.d_model, self.all_head_size)

        self.drop = qc.Dropout(config.drop_attn)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        word_model_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_model_states.size(1)

        if entity_hidden_states is None:
            concat_hidden_states = word_model_states
        else:
            concat_hidden_states = torch.cat([word_model_states, entity_hidden_states], dim=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))

        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # compute query vectors using word-word (w2w), word-entity (w2e), entity-word (e2w), entity-entity (e2e)
            # query layers
            w2w_query_layer = self.transpose_for_scores(self.query(word_model_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_model_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            # compute w2w, w2e, e2w, and e2e key vectors used with the query vectors computed above
            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            # compute attention scores based on the dot product between the query and key vectors
            w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
            w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
            e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
            e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
            entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
            attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        else:
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LukeModel forward() function)
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
        context_layer = context_layer.view(*new_context_layer_shape)

        output_word_model_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_model_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_model_states, output_entity_hidden_states)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LukeSelfOutput(qc.Module):
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
        self.self = LukeSelfAttention(config)
        self.output = LukeSelfOutput(config)

    def forward(
        self,
        word_model_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_model_states.size(1)
        self_outputs = self.self(
            word_model_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_model_states
        else:
            concat_self_outputs = torch.cat(self_outputs[:2], dim=1)
            concat_hidden_states = torch.cat([word_model_states, entity_hidden_states], dim=1)

        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # add attns if we output them
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LukeIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LukeOutput(qc.Module):
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


class Layer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = LukeIntermediate(config)
        self.output = LukeOutput(config)

    def forward(
        self,
        word_model_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_model_states.size(1)

        self_attention_outputs = self.attention(
            word_model_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = torch.cat(self_attention_outputs[:2], dim=1)

        outputs = self_attention_outputs[2:]  # add self attns if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            concat_attention_output,
        )
        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        outputs = (word_layer_output, entity_layer_output) + outputs

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
        word_model_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_word_model_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_model_states = all_word_model_states + (word_model_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    word_model_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    word_model_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            word_model_states = layer_outputs[0]

            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_word_model_states = all_word_model_states + (word_model_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    word_model_states,
                    all_word_model_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        return BaseLukeModelOutput(
            y=word_model_states,
            hiddens=all_word_model_states,
            attns=all_self_attentions,
            entity_last_hidden_state=entity_hidden_states,
            entity_hidden_states=all_entity_hidden_states,
        )


class EntityPredictionHeadTransform(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.entity_emb_size)
        self.act = qu.activation(cfg.act)
        self.norm = qc.LayerNorm(cfg.entity_emb_size, eps=cfg.eps)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        y = self.norm(y)
        return y


class EntityPredictionHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = qc.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hiddens):
        hiddens = self.transform(hiddens)
        hiddens = self.decoder(hiddens) + self.bias

        return hiddens


class Model(PreTrained):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = Encoder(config)

        self.pooler = Pool(config) if add_pooling_layer else None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if entity_ids is not None:
            entity_seq_length = entity_ids.size(1)
            if entity_attention_mask is None:
                entity_attention_mask = torch.ones((batch_size, entity_seq_length), device=device)
            if entity_token_type_ids is None:
                entity_token_type_ids = torch.zeros(
                    (batch_size, entity_seq_length), dtype=torch.long, device=device
                )
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)

        # First, compute word embeddings
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Second, compute extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, entity_attention_mask
        )

        # Third, compute entity embeddings and concatenate with word embeddings
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(
                entity_ids, entity_position_ids, entity_token_type_ids
            )

        # Fourth, send embeddings through the model
        encoder_outputs = self.encoder(
            word_embedding_output,
            entity_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Fifth, get the output. LukeModel outputs the same as BertModel, namely sequence_output of shape (batch_size, seq_len, d_model)
        sequence_output = encoder_outputs[0]

        # Sixth, we compute the pooled_output, word_sequence_output and entity_sequence_output based on the sequence_output
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseLukeModelOutputWithPooling(
            y=sequence_output,
            pools=pooled_output,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            entity_last_hidden_state=encoder_outputs.entity_last_hidden_state,
            entity_hidden_states=encoder_outputs.entity_hidden_states,
        )

    def get_extended_attention_mask(
        self,
        word_attention_mask,
        entity_attention_mask,
    ):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    return incremental_indices.long() + padding_idx


@dataclass
class WithLoss(ModelOutput):
    loss = None
    mlm_loss = None
    mep_loss = None
    logits = None
    entity_logits = None
    hiddens = None
    entity_hidden_states = None
    attns = None


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(add_pool=False, **kw)
        self.proj = Masked(cfg.d_model, eps=1e-12, **kw)
        self.ent_proj = EntityPredictionHead(**kw)

    def forward_masked(self, x, labels=None, ent_labels=None, **kw):
        ys = self.model(x, **kw)
        y = self.proj(ys[0])
        loss, mlm = None, None
        if labels is not None:
            loss = mlm = nn.CrossEntropyLoss()(y.view(-1, self.cfg.s_vocab), labels.view(-1))
        mep = None
        y2 = self.ent_proj(ys.entity_last_hidden_state)
        if ent_labels is not None:
            mep = nn.CrossEntropyLoss()(y2.view(-1, self.cfg.ent_s_vocab), ent_labels.view(-1))
            loss = mep if loss is None else loss + mep
        ys = (y, y2) + ys[1:] + (loss, mlm, mep)
        return WithLoss(*ys)


class LukeForEntityClassification(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.luke = Model(config)

        self.n_labels = config.n_labels
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model, config.n_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        feature_vector = outputs.entity_last_hidden_state[:, 0, :]
        feature_vector = self.drop(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).type_as(logits)
                )

        if not return_dict:
            output = (
                logits,
                outputs.hiddens,
                outputs.entity_hidden_states,
                outputs.attns,
            )
            return ((loss,) + output) if loss is not None else output

        return EntityClassificationOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            entity_hidden_states=outputs.entity_hidden_states,
            attns=outputs.attns,
        )


class LukeForEntityPairClassification(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.luke = Model(config)

        self.n_labels = config.n_labels
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model * 2, config.n_labels, False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        feature_vector = torch.cat(
            [outputs.entity_last_hidden_state[:, 0, :], outputs.entity_last_hidden_state[:, 1, :]],
            dim=1,
        )
        feature_vector = self.drop(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).type_as(logits)
                )

        if not return_dict:
            output = (
                logits,
                outputs.hiddens,
                outputs.entity_hidden_states,
                outputs.attns,
            )
            return ((loss,) + output) if loss is not None else output

        return EntityPairClassificationOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            entity_hidden_states=outputs.entity_hidden_states,
            attns=outputs.attns,
        )


class LukeForEntitySpanClassification(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.luke = Model(config)

        self.n_labels = config.n_labels
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model * 3, config.n_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        entity_start_positions=None,
        entity_end_positions=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        d_model = outputs.y.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, d_model)
        start_states = torch.gather(outputs.y, -2, entity_start_positions)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, d_model)
        end_states = torch.gather(outputs.y, -2, entity_end_positions)
        feature_vector = torch.cat(
            [start_states, end_states, outputs.entity_last_hidden_state], dim=2
        )

        feature_vector = self.drop(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 2, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 2:
                loss = F.cross_entropy(logits.view(-1, self.n_labels), labels.view(-1))
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).type_as(logits)
                )

        if not return_dict:
            output = (
                logits,
                outputs.hiddens,
                outputs.entity_hidden_states,
                outputs.attns,
            )
            return ((loss,) + output) if loss is not None else output

        return EntitySpanClassificationOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            entity_hidden_states=outputs.entity_hidden_states,
            attns=outputs.attns,
        )
