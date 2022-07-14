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

import copy
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
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.canine import PreTrained


log = logging.get_logger(__name__)

LIST = ["google/canine-s", "google/canine-r"]

_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]


class CanineEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        shard_embedding_size = config.d_model // config.num_hash_functions
        for i in range(config.num_hash_functions):
            name = f"HashBucketCodepointEmbedder_{i}"
            setattr(self, name, qc.Embed(config.num_hash_buckets, shard_embedding_size))
        self.char_position_embeddings = qc.Embed(config.num_hash_buckets, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)))
        self.pos_type = getattr(config, "pos_type", "absolute")

    def _hash_bucket_tensors(self, input_ids, num_hashes, num_buckets):
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        primes = _PRIMES[:num_hashes]
        result_tensors = []
        for prime in primes:
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def _embed_hash_buckets(self, input_ids, d_embed, num_hashes, num_buckets):
        if d_embed % num_hashes != 0:
            raise ValueError(f"Expected `d_embed` ({d_embed}) % `num_hashes` ({num_hashes}) == 0")
        hash_bucket_tensors = self._hash_bucket_tensors(
            input_ids, num_hashes=num_hashes, num_buckets=num_buckets
        )
        embedding_shards = []
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            name = f"HashBucketCodepointEmbedder_{i}"
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)
        return torch.cat(embedding_shards, dim=-1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
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
            inputs_embeds = self._embed_hash_buckets(
                input_ids,
                self.config.d_model,
                self.config.num_hash_functions,
                self.config.num_hash_buckets,
            )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.pos_type == "absolute":
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class CharactersToMolecules(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = qc.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.downsampling_rate,
            stride=config.downsampling_rate,
        )
        self.act = qu.activation(config.act)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)

    def forward(self, char_encoding):
        cls_encoding = char_encoding[:, 0:1, :]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.act(downsampled)
        downsampled_truncated = downsampled[:, 0:-1, :]
        result = torch.cat([cls_encoding, downsampled_truncated], dim=1)
        result = self.norm(result)
        return result


class ConvProjection(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv = qc.Conv1d(
            in_channels=config.d_model * 2,
            out_channels=config.d_model,
            kernel_size=config.upsampling_kernel_size,
            stride=1,
        )
        self.act = qu.activation(config.act)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)

    def forward(self, inputs, final_seq_char_positions=None):
        inputs = torch.transpose(inputs, 1, 2)
        pad_total = self.config.upsampling_kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad = nn.ConstantPad1d((pad_beg, pad_end), 0)
        result = self.conv(pad(inputs))
        result = torch.transpose(result, 1, 2)
        result = self.act(result)
        result = self.norm(result)
        result = self.drop(result)
        final_char_seq = result
        if final_seq_char_positions is not None:
            raise NotImplementedError("ForMasked is currently not supported")
        else:
            query_seq = final_char_seq
        return query_seq


class CanineSelfAttention(qc.Module):
    def __init__(self, config):
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
        self.pos_type = getattr(config, "pos_type", "absolute")
        if self.pos_type == "relative_key" or self.pos_type == "relative_key_query":
            self.n_pos = config.n_pos
            self.distance_embedding = qc.Embed(2 * config.n_pos - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        from_tensor,
        to_tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(from_tensor)
        key_layer = self.transpose_for_scores(self.key(to_tensor))
        value_layer = self.transpose_for_scores(self.value(to_tensor))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.pos_type == "relative_key" or self.pos_type == "relative_key_query":
            seq_length = from_tensor.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=from_tensor.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=from_tensor.device
            ).view(1, -1)
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
            if attention_mask.ndim == 3:
                # if attention_mask is 3D, do the following:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                attention_mask = (1.0 - attention_mask.float()) * -10000.0
            # Apply the attention mask (precomputed for all layers in CanineModel forward() function)
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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class CanineSelfOutput(qc.Module):
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


class CanineIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


class CanineOutput(qc.Module):
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


class CaninePredictionHeadTransform(qc.Module):
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


class CanineLMPredictionHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = CaninePredictionHeadTransform(config)
        self.decoder = qc.Linear(config.d_model, config.s_vocab, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.s_vocab))
        self.decoder.bias = self.bias

    def forward(self, x):
        y = self.transform(x)
        y = self.decoder(y)
        return y


class CanineOnlyMLMHead(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = CanineLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class Model(PreTrained):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.n_lays = 1
        self.char_embeddings = CanineEmbeddings(config)
        self.initial_char_encoder = Encoder(
            shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )
        self.chars_to_molecules = CharactersToMolecules(config)
        self.encoder = Encoder(config)
        self.projection = ConvProjection(config)
        self.final_char_encoder = Encoder(shallow_config)
        self.pooler = Pool(config) if add_pooling_layer else None
        self.post_init()

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]
        to_seq_length = to_mask.shape[1]
        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()
        broadcast_ones = torch.ones(
            size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device
        )
        mask = broadcast_ones * to_mask
        return mask

    def _downsample_attention_mask(self, char_attention_mask, downsampling_rate):
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))
        pooled_molecule_mask = torch.nn.MaxPool1d(
            kernel_size=downsampling_rate, stride=downsampling_rate
        )(poolable_char_mask.float())
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)
        return molecule_attention_mask

    def _repeat_molecules(self, molecules, char_seq_length):
        rate = self.config.downsampling_rate
        molecules_without_extra_cls = molecules[:, 1:, :]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_molecule,
            # +1 molecule to compensate for truncation.
            repeats=remainder_length + rate,
            dim=-2,
        )

        # `repeated`: [batch_size, char_seq_len, molecule_hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)

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
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
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
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        molecule_attention_mask = self._downsample_attention_mask(
            attention_mask, downsampling_rate=self.config.downsampling_rate
        )
        extended_molecule_attention_mask = self.get_extended_attention_mask(
            molecule_attention_mask, (batch_size, molecule_attention_mask.shape[-1]), device
        )
        head_mask = self.get_head_mask(head_mask, self.config.n_lays)
        input_char_embeddings = self.char_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        char_attention_mask = self._create_3d_attention_mask_from_input_mask(
            input_ids, attention_mask
        )
        init_chars_encoder_outputs = self.initial_char_encoder(
            input_char_embeddings,
            attention_mask=char_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        input_char_encoding = init_chars_encoder_outputs.y
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)
        encoder_outputs = self.encoder(
            init_molecule_encoding,
            attention_mask=extended_molecule_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None
        repeated_molecules = self._repeat_molecules(
            molecule_sequence_output, char_seq_length=input_shape[-1]
        )
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)
        sequence_output = self.projection(concat)
        final_chars_encoder_outputs = self.final_char_encoder(
            sequence_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = final_chars_encoder_outputs.y

        if output_hidden_states:
            deep_encoder_hidden_states = (
                encoder_outputs.hiddens if return_dict else encoder_outputs[1]
            )
            all_hidden_states = (
                all_hidden_states
                + init_chars_encoder_outputs.hiddens
                + deep_encoder_hidden_states
                + final_chars_encoder_outputs.hiddens
            )

        if output_attentions:
            deep_encoder_self_attentions = (
                encoder_outputs.attns if return_dict else encoder_outputs[-1]
            )
            all_self_attentions = (
                all_self_attentions
                + init_chars_encoder_outputs.attns
                + deep_encoder_self_attentions
                + final_chars_encoder_outputs.attns
            )

        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple(v for v in [all_hidden_states, all_self_attentions] if v is not None)
            return output

        return qo.WithPools(
            y=sequence_output,
            pools=pooled_output,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
        )


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.canine = Model(config)
        self.drop = qc.Dropout(config.drop)
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

        outputs = self.canine(
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

        pooled_output = outputs[1]

        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.WithLoss(
            loss=loss,
            logits=reshaped_logits,
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


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class ForTokClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Classifier(**kw)

    forward = qf.forward_tok


class Encoder(qc.Module):
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                Layer(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                )
                for _ in range(config.n_lays)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
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
                )
            else:
                layer_outputs = layer_module(
                    hiddens, attention_mask, layer_head_mask, output_attentions
                )
            hiddens = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)
        if not return_dict:
            return tuple(
                v for v in [hiddens, all_hidden_states, all_self_attentions] if v is not None
            )
        return qo.Base(
            y=hiddens,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
        )


class Layer(qc.Module):
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(
            config,
            local,
            always_attend_to_first_position,
            first_position_attends_to_all,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )
        self.intermediate = CanineIntermediate(config)
        self.output = CanineOutput(config)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
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


class Attention(qc.Module):
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super().__init__()
        self.self = CanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        self.local = local
        assert attend_from_chunk_width >= attend_from_chunk_stride
        assert attend_to_chunk_width >= attend_to_chunk_stride
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        if not self.local:
            self_outputs = self.self(hiddens, hiddens, attention_mask, head_mask, output_attentions)
            attention_output = self_outputs[0]
        else:
            from_seq_length = to_seq_length = hiddens.shape[1]
            from_tensor = to_tensor = hiddens
            from_chunks = []
            if self.first_position_attends_to_all:
                from_chunks.append((0, 1))
                from_start = 1
            else:
                from_start = 0
            for chunk_start in range(from_start, from_seq_length, self.attend_from_chunk_stride):
                chunk_end = min(from_seq_length, chunk_start + self.attend_from_chunk_width)
                from_chunks.append((chunk_start, chunk_end))
            to_chunks = []
            if self.first_position_attends_to_all:
                to_chunks.append((0, to_seq_length))
            for chunk_start in range(0, to_seq_length, self.attend_to_chunk_stride):
                chunk_end = min(to_seq_length, chunk_start + self.attend_to_chunk_width)
                to_chunks.append((chunk_start, chunk_end))
            if len(from_chunks) != len(to_chunks):
                raise ValueError(
                    f"Expected to have same number of `from_chunks` ({from_chunks}) and "
                    f"`to_chunks` ({from_chunks}). Check strides."
                )
            attention_output_chunks = []
            attention_probs_chunks = []
            for (from_start, from_end), (to_start, to_end) in zip(from_chunks, to_chunks):
                from_tensor_chunk = from_tensor[:, from_start:from_end, :]
                to_tensor_chunk = to_tensor[:, to_start:to_end, :]
                attention_mask_chunk = attention_mask[:, from_start:from_end, to_start:to_end]
                if self.always_attend_to_first_position:
                    cls_attention_mask = attention_mask[:, from_start:from_end, 0:1]
                    attention_mask_chunk = torch.cat(
                        [cls_attention_mask, attention_mask_chunk], dim=2
                    )
                    cls_position = to_tensor[:, 0:1, :]
                    to_tensor_chunk = torch.cat([cls_position, to_tensor_chunk], dim=1)
                attention_outputs_chunk = self.self(
                    from_tensor_chunk,
                    to_tensor_chunk,
                    attention_mask_chunk,
                    head_mask,
                    output_attentions,
                )
                attention_output_chunks.append(attention_outputs_chunk[0])
                if output_attentions:
                    attention_probs_chunks.append(attention_outputs_chunk[1])
            attention_output = torch.cat(attention_output_chunks, dim=1)
        attention_output = self.output(attention_output, hiddens)
        outputs = (attention_output,)
        if not self.local:
            outputs = outputs + self_outputs[1:]
        else:
            outputs = outputs + tuple(attention_probs_chunks)
        return outputs
