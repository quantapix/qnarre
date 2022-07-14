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

import abc
import math
import numpy as np
import torch
import torch.utils.checkpoint

from dataclasses import dataclass
from functools import reduce
from operator import __add__
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
from ..prep.config.perceiver import PreTrained


from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...pytorch_utils import (
    apply_chunking_to_forward,
)

log = logging.get_logger(__name__)

LIST = [
    "deepmind/language-perceiver",
]


@dataclass
class PerceiverModelOutput(ModelOutput):
    logits = None
    y = None
    hiddens = None
    attns = None
    crosses = None


@dataclass
class PerceiverDecoderOutput(ModelOutput):
    logits = None
    crosses = None


class PerceiverEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.d_latents))

    def forward(self, batch_size):
        return self.latents.expand(batch_size, -1, -1)  # Thanks, Phil Wang


class PerceiverSelfAttention(qc.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        if qk_channels is None:
            qk_channels = q_dim
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % n_heads != 0:
            raise ValueError(
                f"qk_channels ({qk_channels}) must be divisible by n_heads ({n_heads})."
            )
        if v_channels % n_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by n_heads ({n_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // n_heads
        self.v_channels_per_head = self.v_channels // n_heads

        # Layer normalization
        self.layernorm1 = qc.LayerNorm(q_dim)
        self.layernorm2 = qc.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = qc.Linear(q_dim, qk_channels)
        self.key = qc.Linear(kv_dim, qk_channels)
        self.value = qc.Linear(kv_dim, v_channels)

        self.drop = qc.Dropout(config.drop_attn)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.n_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        hiddens = self.layernorm1(hiddens)
        inputs = self.layernorm2(inputs)
        is_cross_attention = inputs is not None
        queries = self.query(hiddens)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hiddens)
            values = self.value(hiddens)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, n_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.n_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.drop(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PerceiverSelfOutput(qc.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        self.dense = qc.Linear(input_channels, output_channels)

    def forward(self, hiddens):
        hiddens = self.dense(hiddens)
        return hiddens


class Attention(qc.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            n_heads=n_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(
            config, input_channels=self.self.v_channels, output_channels=output_channels
        )
        self.use_query_residual = use_query_residual

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hiddens,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0])
        if self.use_query_residual:
            attention_output = attention_output + hiddens
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class PerceiverMLP(qc.Module):
    def __init__(self, cfg, input_size, widening_factor):
        super().__init__()
        self.dense1 = qc.Linear(input_size, widening_factor * input_size)
        self.act = qu.activation(cfg.act)
        self.dense2 = qc.Linear(widening_factor * input_size, input_size)

    def forward(self, x):
        y = self.dense1(x)
        y = self.act(y)
        y = self.dense2(y)
        return y


class Layer(qc.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            n_heads=n_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = qc.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hiddens,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attns if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class Encoder(qc.Module):
    def __init__(self, config, kv_dim=None):
        super().__init__()
        self.config = config
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        self.cross_attention = Layer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            n_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=kv_dim,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = Layer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                n_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hiddens,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # Apply the cross-attention between the latents (hiddens) and inputs:
        layer_outputs = self.cross_attention(
            hiddens,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hiddens = layer_outputs[0]

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hiddens,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hiddens,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hiddens = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

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


class Model(PreTrained):
    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor=None,
        output_postprocessor=None,
    ):
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = Encoder(
            config,
            kv_dim=input_preprocessor.num_channels
            if input_preprocessor is not None
            else config.d_model,
        )
        self.decoder = decoder

    def forward(
        self,
        inputs,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
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

        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            if inputs.size()[-1] != self.config.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model: {self.config.d_model}. "
                    "Make sure to set config.d_model appropriately."
                )

        batch_size, seq_length, _ = inputs.size()
        device = inputs.device

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        # Make the attention mask broadcastable to [batch_size, n_heads, seq_length, seq_length]
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.num_blocks * self.config.num_self_attends_per_block
        )

        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            if subsampled_output_points is not None:
                output_modality_sizes = {
                    "audio": subsampled_output_points["audio"].shape[0],
                    "image": subsampled_output_points["image"].shape[0],
                    "label": 1,
                }
            else:
                output_modality_sizes = None
            decoder_query = self.decoder.decoder_query(
                inputs,
                modality_sizes,
                inputs_without_pos,
                subsampled_points=subsampled_output_points,
            )
            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            # add cross-attns of decoder
            if output_attentions and decoder_outputs.crosses is not None:
                if return_dict:
                    encoder_outputs.crosses = encoder_outputs.crosses + decoder_outputs.crosses
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.crosses

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)

        if not return_dict:
            if logits is not None:
                return (logits, sequence_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return PerceiverModelOutput(
            logits=logits,
            y=sequence_output,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            crosses=encoder_outputs.crosses,
        )


class ForMasked(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        text_preprocessor = PerceiverTextPreprocessor(config)
        trainable_position_encoding_kwargs_decoder = dict(
            num_channels=text_preprocessor.num_channels, index_dims=config.n_pos
        )
        self.perceiver = Model(
            config,
            input_preprocessor=text_preprocessor,
            decoder=PerceiverBasicDecoder(
                config,
                output_num_channels=config.d_latents,
                output_index_dims=config.n_pos,  # we need to define the seq_len of the inputs beforehand
                num_channels=text_preprocessor.num_channels,
                qk_channels=8 * 32,
                v_channels=text_preprocessor.num_channels,
                n_heads=8,
                use_query_residual=False,
                final_project=False,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
            ),
        )
        self.embedding_decoder = PerceiverEmbeddingDecoder(config)
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        input_ids=None,
    ):
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.embedding_decoder(
            outputs.logits if return_dict else outputs[0],
            embedding_layer=self.perceiver.input_preprocessor.embeddings,
        )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.s_vocab), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return qo.LossCrosses(
            loss=masked_lm_loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        kw2 = dict(num_channels=cfg.d_latents, index_dims=1)
        self.model = Model(
            input_preprocessor=PerceiverTextPreprocessor(cfg),
            decoder=PerceiverClassificationDecoder(
                cfg,
                num_channels=cfg.d_latents,
                trainable_position_encoding_kwargs=kw2,
                use_query_residual=True,
            ),
            **kw,
        )
        self.proj = Classifier(**kw)

    forward = qf.forward_seq


class PerceiverForImageClassificationLearned(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        trainable_position_encoding_kwargs_preprocessor = dict(
            num_channels=256, index_dims=config.image_size**2
        )
        trainable_position_encoding_kwargs_decoder = dict(
            num_channels=config.d_latents, index_dims=1
        )

        self.n_labels = config.n_labels
        self.perceiver = Model(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.n_labels == 1:
                    self.config.problem_type = "regression"
                elif self.n_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCrosses(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


class PerceiverForImageClassificationFourier(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            concat_pos=True, max_resolution=(224, 224), num_bands=64, sine_only=False
        )
        trainable_position_encoding_kwargs_decoder = dict(
            num_channels=config.d_latents, index_dims=1
        )

        self.n_labels = config.n_labels
        self.perceiver = Model(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="pixels",
                spatial_downsample=1,
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.n_labels == 1:
                    self.config.problem_type = "regression"
                elif self.n_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCrosses(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


class PerceiverForImageClassificationConvProcessing(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            concat_pos=True, max_resolution=(56, 56), num_bands=64, sine_only=False
        )
        trainable_position_encoding_kwargs_decoder = dict(
            num_channels=config.d_latents, index_dims=1
        )

        self.n_labels = config.n_labels
        self.perceiver = Model(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv",
                spatial_downsample=1,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.n_labels == 1:
                    self.config.problem_type = "regression"
                elif self.n_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCrosses(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


class PerceiverForOpticalFlow(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            num_bands=64,
            max_resolution=config.train_size,
            sine_only=False,
            concat_pos=True,
        )
        fourier_position_encoding_kwargs_decoder = dict(
            concat_pos=True, max_resolution=config.train_size, num_bands=64, sine_only=False
        )

        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="patches",
            spatial_downsample=1,
            conv_after_patching=True,
            conv_after_patching_in_channels=54,
            temporal_downsample=2,
            position_encoding_type="fourier",
            # position_encoding_kwargs
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )

        self.perceiver = Model(
            config,
            input_preprocessor=image_preprocessor,
            decoder=PerceiverOpticalFlowDecoder(
                config,
                num_channels=image_preprocessor.num_channels,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                # decoder kw
                use_query_residual=False,
                output_num_channels=2,
                # We query the decoder using the first frame features
                # rather than a standard decoder position encoding.
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Optical flow training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCrosses(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


class PerceiverForMultimodalAutoencoding(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        n_audio_samples = config.num_frames * config.audio_samples_per_frame

        input_preprocessor = PerceiverMultimodalPreprocessor(
            min_padding_size=4,
            modalities={
                "audio": PerceiverAudioPreprocessor(
                    config,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    prep_type="patches",
                    samples_per_patch=config.samples_per_patch,
                ),
                "image": PerceiverImagePreprocessor(
                    config,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=32,
                        max_resolution=(config.num_frames, config.image_size, config.image_size),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    prep_type="patches",
                    spatial_downsample=4,
                    temporal_downsample=1,
                ),
                "label": PerceiverOneHotPreprocessor(config),
            },
            mask_probs={"image": 0.0, "audio": 0.0, "label": 1.0},
        )

        image_decoder = PerceiverBasicVideoAutoencodingDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            output_shape=config.output_shape,
            output_num_channels=512,
            use_query_residual=False,
            position_encoding_only=True,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs=dict(
                num_bands=32,
                max_resolution=(config.num_frames, config.image_size, config.image_size),
                sine_only=False,
                concat_pos=True,
            ),
        )

        decoder = PerceiverMultimodalDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            # Modality specific decoders are used ONLY to generate queries.
            # All modalties are decoded together using a unified decoder.
            modalities={
                "audio": PerceiverBasicDecoder(
                    config,
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    output_index_dims=(n_audio_samples // config.samples_per_patch,),
                    output_num_channels=512,
                    use_query_residual=False,
                    position_encoding_only=True,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                ),
                "image": image_decoder,
                "label": PerceiverClassificationDecoder(
                    config,
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    use_query_residual=False,
                    position_encoding_only=True,
                    position_encoding_type="trainable",
                    trainable_position_encoding_kwargs=dict(
                        num_channels=1024,
                        index_dims=1,
                    ),
                ),
            },
            num_outputs=None,
            output_num_channels=512,
            use_query_residual=False,
        )

        output_postprocessor = PerceiverMultimodalPostprocessor(
            modalities={
                "audio": PerceiverAudioPostprocessor(config, in_channels=512),
                "image": PerceiverProjectionPostprocessor(in_channels=512, out_channels=3),
                "label": PerceiverClassificationPostprocessor(config, in_channels=512),
            }
        )

        self.perceiver = Model(
            config,
            input_preprocessor=input_preprocessor,
            decoder=decoder,
            output_postprocessor=output_postprocessor,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            subsampled_output_points=subsampled_output_points,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Multimodal autoencoding training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return qo.LossCrosses(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
            crosses=outputs.crosses,
        )


def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
):
    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    positions_projection = (
        qc.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()
    )

    return output_pos_enc, positions_projection


class PerceiverAbstractDecoder(qc.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_query_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, query_mask=None):
        raise NotImplementedError


class PerceiverProjectionDecoder(PerceiverAbstractDecoder):
    def __init__(self, config):
        super().__init__()
        self.classifier = qc.Linear(config.d_latents, config.n_labels)

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return None

    def forward(self, query, z, query_mask=None):
        z = torch.mean(z, dim=1)
        logits = self.classifier(z)
        return logits


class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config,
        output_num_channels,
        position_encoding_type="trainable",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims=None,
        num_channels=128,
        subsampled_index_dims=None,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        widening_factor=1,
        use_query_residual=False,
        concat_preprocessed_input=False,
        final_project=True,
        position_encoding_only=False,
        **position_encoding_kwargs,
    ):
        super().__init__()

        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True
        if not self.position_encoding_only:
            self.decoding_cross_attention = Layer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                n_heads=n_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            self.final_layer = (
                qc.Linear(num_channels, output_num_channels) if final_project else nn.Identity()
            )

    @property
    def num_query_channels(self):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot construct decoder queries when position_encoding_type is set to none"
            )
        if subsampled_points is not None:
            indices = list(
                torch.from_numpy(x)
                for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)
            )
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    index_dims, batch_size, device=inputs.device
                )

            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError(
                    "Value is required for inputs_without_pos if concat_preprocessed_input is True"
                )
            pos_emb = torch.cat([inputs_without_pos, pos_emb], div=-1)

        return pos_emb

    def forward(self, query, z, query_mask=None, output_attentions=False):
        crosses = () if output_attentions else None

        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        if output_attentions:
            crosses = crosses + (layer_outputs[1],)

        logits = self.final_layer(output)

        return PerceiverDecoderOutput(logits=logits, crosses=crosses)


class PerceiverClassificationDecoder(PerceiverAbstractDecoder):
    def __init__(self, config, **decoder_kwargs):
        super().__init__()

        self.n_labels = config.n_labels
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=self.n_labels,
            output_index_dims=1,  # Predict a single logit array.
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self):
        return self.decoder.num_query_channels

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        logits = decoder_outputs.logits[:, 0, :]

        return PerceiverDecoderOutput(logits=logits, crosses=decoder_outputs.crosses)


class PerceiverOpticalFlowDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config,
        output_image_shape,
        output_num_channels=2,
        rescale_factor=100.0,
        **decoder_kwargs,
    ):
        super().__init__()

        self.output_image_shape = output_image_shape
        self.output_num_channels = output_num_channels
        self.rescale_factor = rescale_factor
        self.decoder = PerceiverBasicDecoder(
            config, output_num_channels=output_num_channels, **decoder_kwargs
        )

    @property
    def num_query_channels(self):
        return self.decoder.num_query_channels

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        return inputs

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        # Output flow and rescale.
        preds /= self.rescale_factor
        preds = preds.reshape([preds.shape[0]] + list(self.output_image_shape) + [preds.shape[-1]])
        return PerceiverDecoderOutput(logits=preds, crosses=decoder_outputs.crosses)


class PerceiverBasicVideoAutoencodingDecoder(PerceiverAbstractDecoder):
    def __init__(self, config, output_shape, position_encoding_type, **decoder_kwargs):
        super().__init__()
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # Build the decoder components:
        self.output_shape = output_shape
        self.output_num_channels = decoder_kwargs["output_num_channels"]

        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=self.output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self):
        return self.decoder.num_query_channels

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def forward(self, query, z, query_mask=None):
        decoder_outputs = self.decoder(query, z)
        logits = decoder_outputs.logits

        logits = torch.reshape(logits, self.output_shape + [logits.shape[-1]])
        return PerceiverDecoderOutput(logits=logits, crosses=decoder_outputs.crosses)


def restructure(modality_sizes, inputs):
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config,
        modalities,
        num_outputs,
        output_num_channels,
        min_padding_size=2,
        subsampled_index_dims=None,
        **decoder_kwargs,
    ):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.subsampled_index_dims = subsampled_index_dims
        self.min_padding_size = min_padding_size
        self.output_num_channels = output_num_channels
        self.num_outputs = num_outputs
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            num_channels=self.num_query_channels,
            **decoder_kwargs,
        )
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.randn(1, self.num_query_channels - decoder.num_query_channels)
                )
                for modality, decoder in modalities.items()
            }
        )

    @property
    def num_query_channels(self):
        max_channel_size = max(decoder.num_query_channels for _, decoder in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def decoder_query(
        self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None
    ):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)

        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()

        decoder_queries = dict()
        for modality, decoder in self.modalities.items():
            # Get input_without_pos for this modality if it exists.
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            query = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )
            decoder_queries[modality] = query

        # Pad all queries with trainable position encodings to make them have the same channels

        def embed(modality, x):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = self.padding[modality]
            pos = torch.broadcast_to(
                pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]]
            )
            return torch.cat([x, pos], dim=2)

        # Apply a predictable ordering to the modalities
        return torch.cat(
            [
                embed(modality, decoder_queries[modality])
                for modality in sorted(self.modalities.keys())
            ],
            dim=1,
        )

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        return decoder_outputs


# Below: IO pre- and post-processor classes for Perceiver.
def space_to_depth(frames, temporal_block_size=1, spatial_block_size=1):
    if len(frames.shape) == 4:
        batch_size, num_channels, height, width = frames.shape
        # split up dimensions (height by spatial_block_size, width by spatial_block_size)
        frames = frames.view(
            batch_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # move blocks to last dimension: (batch_size, H//bs, W//bs, bs, bs, C)
        frames = frames.permute(0, 2, 4, 3, 5, 1).contiguous()
        # concatenate blocks along channel dimension: (batch_size, H//bs, W//bs, bs*bs*C)
        frames = frames.view(
            batch_size,
            height // spatial_block_size,
            width // spatial_block_size,
            (spatial_block_size**2) * num_channels,
        )
        return frames
    elif len(frames.shape) == 5:
        batch_size, time, num_channels, height, width = frames.shape
        # split up dimensions (time by temporal_block_size, height by spatial_block_size, width by spatial_block_size)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            temporal_block_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # move blocks to last dimension: (batch_size, T//ts, H//bs, W//bs, ts, bs, bs, C)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        # concatenate blocks along channel dimension: (batch_size, T//ts, H//bs, W//bs, ts*bs*bs*C)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            height // spatial_block_size,
            width // spatial_block_size,
            temporal_block_size * (spatial_block_size**2) * num_channels,
        )
        return frames
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kw):
        super(Conv2dSamePadding, self).__init__(*args, **kw)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(
                __add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]
            )
        )

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Conv2DDownsample(qc.Module):
    def __init__(
        self,
        n_lays=1,
        in_channels=3,
        out_channels=64,
        use_batchnorm=True,
    ):
        super().__init__()

        self.conv = Conv2dSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False
        )
        self.batchnorm = (
            nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False
):
    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution],
        dim=0,
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    def _linspace(n_xels_per_dim):
        return torch.linspace(
            start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32
        )

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)


class PerceiverAbstractPositionEncoding(qc.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def num_dimensions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_size(self, *args, **kw):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError


class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self):
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kw):
        return self._num_channels

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        return position_embeddings


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos


class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self):
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


class AbstractPreprocessor(qc.Module):
    @property
    def num_channels(self):
        raise NotImplementedError()


class PerceiverTextPreprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = qc.Embed(num_embeddings=config.s_vocab, embedding_dim=config.d_model)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)

    @property
    def num_channels(self):
        return self.config.d_model

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)

        seq_length = inputs.shape[1]
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        embeddings = embeddings + self.position_embeddings(position_ids)

        return embeddings, None, None


class PerceiverEmbeddingDecoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.s_vocab = config.s_vocab
        self.bias = nn.Parameter(torch.zeros(self.s_vocab))

    def forward(self, hiddens, embedding_layer):
        batch_size, seq_len, d_model = hiddens.shape
        output = torch.matmul(
            hiddens.reshape([-1, d_model]), embedding_layer.weight.T
        )  # Flatten batch dim
        output = output + self.bias

        return output.reshape([batch_size, seq_len, self.s_vocab])


class PerceiverMultimodalPostprocessor(qc.Module):
    def __init__(self, modalities, input_is_dict=False):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.input_is_dict = input_is_dict

    def forward(self, inputs, pos=None, modality_sizes=None):
        if not self.input_is_dict:
            # Slice up modalities by their sizes.
            if modality_sizes is None:
                raise ValueError("Modality sizes should be specified if input is not a dictionary.")
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)

        outputs = {
            modality: postprocessor(inputs[modality], pos=pos, modality_sizes=None)
            for modality, postprocessor in self.modalities.items()
        }
        return outputs


class PerceiverClassificationPostprocessor(qc.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.classifier = qc.Linear(in_channels, config.n_labels)

    def forward(self, inputs, pos=None, modality_sizes=None):
        logits = self.classifier(inputs)
        return logits[:, 0, :]


class PerceiverAudioPostprocessor(qc.Module):
    def __init__(self, config, in_channels, postproc_type="patches"):
        super().__init__()

        if postproc_type not in ("patches",):  # to be supported: 'conv', 'patches', 'pixels'
            raise ValueError("Invalid postproc_type!")

        # Architecture parameters:
        self.classifier = qc.Linear(in_channels, config.samples_per_patch)

    def forward(self, inputs, pos=None, modality_sizes=None):

        logits = self.classifier(inputs)
        return torch.reshape(logits, [inputs.shape[0], -1])


class PerceiverProjectionPostprocessor(qc.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.classifier = qc.Linear(in_channels, out_channels)

    def forward(self, inputs, pos=None, modality_sizes=None):
        logits = self.classifier(inputs)
        return logits


class PerceiverImagePreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        config,
        prep_type="conv",
        spatial_downsample=4,
        temporal_downsample=1,
        position_encoding_type="fourier",
        in_channels=3,
        out_channels=64,
        conv_after_patching=False,
        conv_after_patching_in_channels=54,  # only relevant when conv_after_patching = True
        conv2d_use_batchnorm=True,
        concat_or_add_pos="concat",
        project_pos_dim=-1,
        **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv."
                )
            self.convnet = Conv2DDownsample(
                in_channels=in_channels,
                n_lays=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )

        elif self.prep_type == "conv1x1":
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        self.conv_after_patches = (
            qc.Linear(conv_after_patching_in_channels, self.out_channels)
            if conv_after_patching
            else nn.Identity()
        )

    @property
    def num_channels(self):
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample**2
                if is_temporal:
                    inp_dim *= self.temporal_downsample

        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs, pos, network_input_is_1d=True):
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        return inputs_with_pos, inputs

    def forward(
        self,
        inputs,
        pos=None,
        network_input_is_1d=True,
    ):
        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

        elif self.prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :,
                    :: self.temporal_downsample,
                    :,
                    :: self.spatial_downsample,
                    :: self.spatial_downsample,
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x C x H x W
            inputs = space_to_depth(
                inputs,
                temporal_block_size=self.temporal_downsample,
                spatial_block_size=self.spatial_downsample,
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = inputs.squeeze(dim=1)

            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # move channels to last dimension, as the _build_network_inputs method below expects this
            if inputs.ndim == 4:
                inputs = torch.moveaxis(inputs, 1, -1)
            elif inputs.ndim == 5:
                inputs = torch.moveaxis(inputs, 2, -1)
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos


class PerceiverOneHotPreprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def num_channels(self):
        return self.config.n_labels

    def forward(
        self,
        inputs,
        pos=None,
        network_input_is_1d=True,
    ):
        inputs = inputs[:, None, :]
        return inputs, None, inputs


class PerceiverAudioPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        config,
        prep_type="patches",
        samples_per_patch=96,
        position_encoding_type="fourier",
        concat_or_add_pos="concat",
        out_channels=64,
        project_pos_dim=-1,
        **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("patches",):
            raise ValueError(f"Prep_type {prep_type} is invalid, can only be 'patches'.")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(
                f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'."
            )

        self.samples_per_patch = samples_per_patch
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim

        # Position embeddings
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self):
        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim
        return self.samples_per_patch + pos_dim

    def _build_network_inputs(self, inputs, pos):
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        pos_enc = self.positions_projection(pos_enc)

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(
        self,
        inputs,
        pos=None,
        network_input_is_1d=True,
    ):
        inputs = torch.reshape(inputs, [inputs.shape[0], -1, self.samples_per_patch])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos


class PerceiverMultimodalPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        modalities,
        mask_probs=None,
        min_padding_size=2,
    ):
        super().__init__()
        self.modalities = modalities
        self.min_padding_size = min_padding_size
        self.mask_probs = mask_probs if mask_probs is not None else dict()
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.randn(1, self.num_channels - preprocessor.num_channels)
                )
                for modality, preprocessor in modalities.items()
            }
        )
        self.mask = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_channels))
                for modality, _ in self.mask_probs.items()
            }
        )

    @property
    def num_channels(self):
        max_channel_size = max(processor.num_channels for _, processor in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def forward(
        self,
        inputs,
        pos=None,
        network_input_is_1d=True,
    ):
        padded = {}
        modality_sizes = {}
        inputs_without_pos = {}
        for modality, preprocessor in self.modalities.items():
            # preprocess each modality using the respective preprocessor.
            output, _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d
            )

            # pad to the same common_channel_size.
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)

            padding = torch.broadcast_to(
                pos_enc,
                [batch_size, num_samples, self.num_channels - num_channels],
            )
            output_padded = torch.cat([output, padding], dim=2)

            # mask if required
            if modality in self.mask_probs:
                msk = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(msk.device)
                output_padded = (1 - mask) * output_padded + mask * msk

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]

        # Apply a predictable ordering to the modalities
        padded_ls = [padded[k] for k in sorted(padded.keys())]

        # Finally, concatenate along the time dimension
        final_inputs = torch.cat(padded_ls, dim=1)

        return final_inputs, modality_sizes, inputs_without_pos
