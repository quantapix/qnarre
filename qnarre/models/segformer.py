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

from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.bert import PreTrained

import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


log = logging.get_logger(__name__)


LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
]


class SegFormerImageClassifierOutput(ImageClassifierOutput):
    loss = None
    logits = None
    hiddens = None
    attns = None


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(x, drop_prob: float = 0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Segformer
class SegformerDropPath(qc.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SegformerOverlapPatchEmbeddings(qc.Module):
    def __init__(self, patch_size, stride, num_channels, d_model):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            d_model,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = qc.LayerNorm(d_model)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerEfficientSelfAttention(qc.Module):
    def __init__(self, config, d_model, n_heads, sequence_reduction_ratio):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"The hidden size ({self.d_model}) is not a multiple of the number of attention "
                f"heads ({self.n_heads})"
            )

        self.attention_head_size = int(self.d_model / self.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.query = qc.Linear(self.d_model, self.all_head_size)
        self.key = qc.Linear(self.d_model, self.all_head_size)
        self.value = qc.Linear(self.d_model, self.all_head_size)

        self.drop = qc.Dropout(config.drop_attn)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                d_model,
                d_model,
                kernel_size=sequence_reduction_ratio,
                stride=sequence_reduction_ratio,
            )
            self.layer_norm = qc.LayerNorm(d_model)

    def transpose_for_scores(self, hiddens):
        new_shape = hiddens.size()[:-1] + (self.n_heads, self.attention_head_size)
        hiddens = hiddens.view(*new_shape)
        return hiddens.permute(0, 2, 1, 3)

    def forward(
        self,
        hiddens,
        height,
        width,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hiddens))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hiddens.shape
            # Reshape to (batch_size, num_channels, height, width)
            hiddens = hiddens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # Apply sequence reduction
            hiddens = self.sr(hiddens)
            # Reshape back to (batch_size, seq_len, num_channels)
            hiddens = hiddens.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hiddens = self.layer_norm(hiddens)

        key_layer = self.transpose_for_scores(self.key(hiddens))
        value_layer = self.transpose_for_scores(self.value(hiddens))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.drop(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SegformerSelfOutput(qc.Module):
    def __init__(self, config, d_model):
        super().__init__()
        self.dense = qc.Linear(d_model, d_model)
        self.drop = qc.Dropout(config.drop)

    def forward(self, hiddens, input_tensor):
        hiddens = self.dense(hiddens)
        hiddens = self.drop(hiddens)
        return hiddens


class Attention(qc.Module):
    def __init__(self, config, d_model, n_heads, sequence_reduction_ratio):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            d_model=d_model,
            n_heads=n_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, d_model=d_model)


class SegformerDWConv(qc.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hiddens, height, width):
        batch_size, seq_len, num_channels = hiddens.shape
        hiddens = hiddens.transpose(1, 2).view(batch_size, num_channels, height, width)
        hiddens = self.dwconv(hiddens)
        hiddens = hiddens.flatten(2).transpose(1, 2)

        return hiddens


class SegformerMixFFN(qc.Module):
    def __init__(self, cfg, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = qc.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.act = qu.activation(cfg.act)
        self.dense2 = qc.Linear(hidden_features, out_features)
        self.drop = qc.Dropout(cfg.drop)

    def forward(self, x, height, width):
        y = self.dense1(x)
        y = self.dwconv(y, height, width)
        y = self.act(y)
        y = self.drop(y)
        y = self.dense2(y)
        y = self.drop(y)
        return y


class Layer(qc.Module):
    def __init__(
        self,
        config,
        d_model,
        n_heads,
        drop_path,
        sequence_reduction_ratio,
        mlp_ratio,
    ):
        super().__init__()
        self.layer_norm_1 = qc.LayerNorm(d_model)
        self.attention = Attention(
            config,
            d_model=d_model,
            n_heads=n_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = qc.LayerNorm(d_model)
        mlp_hidden_size = int(d_model * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=d_model, hidden_features=mlp_hidden_size)

    def forward(self, hiddens, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hiddens),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attns if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hiddens = attention_output + hiddens

        mlp_output = self.mlp(self.layer_norm_2(hiddens), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hiddens

        outputs = (layer_output,) + outputs

        return outputs


class Encoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    d_model=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    Layer(
                        config,
                        d_model=config.hidden_sizes[i],
                        n_heads=config.n_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [qc.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hiddens = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hiddens, height, width = embedding_layer(hiddens)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hiddens, height, width, output_attentions)
                hiddens = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hiddens = norm_layer(hiddens)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hiddens = (
                    hiddens.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )
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


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(config)
        self.post_init()

    def forward(
        self,
        pixel_values,
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

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return qo.Base(
            y=sequence_output,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
        )


class SegformerForImageClassification(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.n_labels = config.n_labels
        self.segformer = Model(config)
        self.classifier = qc.Linear(config.hidden_sizes[-1], config.n_labels)

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, d_model)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class SegformerMLP(qc.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.proj = qc.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hiddens):
        hiddens = hiddens.flatten(2).transpose(1, 2)
        hiddens = self.proj(hiddens)
        return hiddens


class SegformerDecodeHead(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.drop = qc.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.n_labels, kernel_size=1)

        self.config = config

    def forward(self, enc_hiddens):
        batch_size = enc_hiddens[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(enc_hiddens, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=enc_hiddens[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hiddens = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hiddens = self.batch_norm(hiddens)
        hiddens = self.act(hiddens)
        hiddens = self.drop(hiddens)

        # logits are of shape (batch_size, n_labels, height/4, width/4)
        logits = self.classifier(hiddens)

        return logits


class SegformerForSemanticSegmentation(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = Model(config)
        self.decode_head = SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        enc_hiddens = outputs.hiddens if return_dict else outputs[1]

        logits = self.decode_head(enc_hiddens)

        loss = None
        if labels is not None:
            if self.config.n_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hiddens=outputs.hiddens if output_hidden_states else None,
            attns=outputs.attns,
        )
