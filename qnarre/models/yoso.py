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
import os

import torch
import torch.utils.checkpoint

from torch.nn import functional as F
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import forward as qf
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.ffnet import Classifier, FFNet, Masker, Pool
from ..prep.config.yoso import PreTrained

from ...pytorch_utils import (
    apply_chunking_to_forward,
)

log = logging.get_logger(__name__)

LIST = [
    "uw-madison/yoso-4096",
]


def load_cuda_kernels():
    global lsh_cumulation
    try:
        from torch.utils.cpp_extension import load

        def append_root(files):
            src_folder = os.path.dirname(os.path.realpath(__file__))
            return [os.path.join(src_folder, file) for file in files]

        src_files = append_root(
            [
                "fast_lsh_cumulation_torch.cpp",
                "fast_lsh_cumulation.cu",
                "fast_lsh_cumulation_cuda.cu",
            ]
        )

        load("fast_lsh_cumulation", src_files, verbose=True)

        import fast_lsh_cumulation as lsh_cumulation

        return True
    except Exception:
        lsh_cumulation = None
        return False


def to_contiguous(xs):
    if isinstance(xs, list):
        ys = []
        for x in xs:
            if not x.is_contiguous():
                x = x.contiguous()
            ys.append(x)
        return ys
    else:
        if not xs.is_contiguous():
            xs = xs.contiguous()
        return xs


def normalize(xs):
    if type(xs) is list:
        ys = []
        for x in xs:
            ys.append(F.normalize(x, p=2, dim=-1))
        return ys
    else:
        return F.normalize(xs, p=2, dim=-1)


def hashing(q, k, num_hash, hash_len):
    if len(q.size()) != 3:
        raise ValueError("Query has incorrect size.")
    if len(k.size()) != 3:
        raise ValueError("Key has incorrect size.")
    rmat = torch.randn(q.size(0), q.size(2), num_hash * hash_len, device=q.device)
    raise_pow = 2 ** torch.arange(hash_len, device=q.device)
    q = torch.matmul(q, rmat).reshape(q.size(0), q.size(1), num_hash, hash_len)
    k = torch.matmul(k, rmat).reshape(k.size(0), k.size(1), num_hash, hash_len)
    q = (q > 0).int()
    k = (k > 0).int()
    y = torch.sum(q * raise_pow, dim=-1)
    y = torch.sum(k * raise_pow, dim=-1)
    return y.int(), y.int()


class YosoCumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        hash_code_len = config["hash_code_len"]
        expectation = (
            1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi
        ) ** hash_code_len
        expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
        cumulation_value = torch.matmul(expectation, value)
        ctx.save_for_backward(query_mask, key_mask, expectation, query, key, value)
        ctx.config = config
        return cumulation_value

    @staticmethod
    def backward(ctx, grad):
        grad = to_contiguous(grad)
        query_mask, key_mask, expectation, query, key, value = ctx.saved_tensors
        config = ctx.config
        hash_code_len = config["hash_code_len"]
        weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
        grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
        grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
        grad_value = torch.matmul(expectation.transpose(-1, -2), grad)
        return None, None, grad_query, grad_key, grad_value, None


class YosoLSHCumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        if query_mask.size(0) != key_mask.size(0):
            raise ValueError("Query mask and Key mask differ in sizes in dimension 0")
        if query_mask.size(0) != query.size(0):
            raise ValueError("Query mask and Query differ in sizes in dimension 0")
        if query_mask.size(0) != key.size(0):
            raise ValueError("Query mask and Key differ in sizes in dimension 0")
        if query_mask.size(0) != value.size(0):
            raise ValueError("Query mask and Value mask differ in sizes in dimension 0")
        if key.size(1) != value.size(1):
            raise ValueError("Key and Value differ in sizes in dimension 1")
        if query.size(2) != key.size(2):
            raise ValueError("Query and Key differ in sizes in dimension 2")
        query_mask, key_mask, query, key, value = to_contiguous(
            [query_mask, key_mask, query, key, value]
        )
        use_cuda = query_mask.is_cuda
        num_hash = config["num_hash"]
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2**hash_code_len)
        if config["use_fast_hash"]:
            query_hash_code, key_hash_code = lsh_cumulation.fast_hash(
                query_mask, query, key_mask, key, num_hash, hash_code_len, use_cuda, 1
            )
        else:
            query_hash_code, key_hash_code = hashing(query, key, num_hash, hash_code_len)
        cumulation_value = lsh_cumulation.lsh_cumulation(
            query_mask,
            query_hash_code,
            key_mask,
            key_hash_code,
            value,
            hashtable_capacity,
            use_cuda,
            1,
        )
        ctx.save_for_backward(
            query_mask, key_mask, query_hash_code, key_hash_code, query, key, value
        )
        ctx.config = config
        return cumulation_value

    @staticmethod
    def backward(ctx, grad):
        grad = to_contiguous(grad)

        query_mask, key_mask, query_hash_code, key_hash_code, query, key, value = ctx.saved_tensors
        config = ctx.config

        use_cuda = grad.is_cuda
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2**hash_code_len)

        if config["lsh_backward"]:
            grad_value = lsh_cumulation.lsh_cumulation(
                key_mask,
                key_hash_code,
                query_mask,
                query_hash_code,
                grad,
                hashtable_capacity,
                use_cuda,
                1,
            )
            grad_query = lsh_cumulation.lsh_weighted_cumulation(
                query_mask,
                query_hash_code,
                grad,
                key_mask,
                key_hash_code,
                value,
                (hash_code_len / 2) * key,
                hashtable_capacity,
                use_cuda,
                4,
            )
            grad_key = lsh_cumulation.lsh_weighted_cumulation(
                key_mask,
                key_hash_code,
                value,
                query_mask,
                query_hash_code,
                grad,
                (hash_code_len / 2) * query,
                hashtable_capacity,
                use_cuda,
                4,
            )
        else:
            expectation = (
                1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi
            ) ** hash_code_len
            expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
            weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
            grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
            grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
            grad_value = torch.matmul(expectation.transpose(-1, -2), grad)

        return None, None, grad_query, grad_key, grad_value, None


# Copied from transformers.models.nystromformer.modeling_nystromformer.NystromformerEmbeddings
class YosoEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.tok = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.pos = qc.Embed(config.n_pos + 2, config.d_model)
        self.typ = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.drop = qc.Dropout(config.drop)
        self.register_buffer("position_ids", torch.arange(config.n_pos).expand((1, -1)) + 2)
        self.pos_type = getattr(config, "pos_type", "absolute")
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(
                    self.position_ids.size(), dtype=torch.long, device=self.position_ids.device
                ),
                persistent=False,
            )

    def forward(self, x=None, typ=None, pos=None, inputs_embeds=None):
        if x is not None:
            s = x.size()
        else:
            s = inputs_embeds.size()[:-1]
        seq_length = s[1]
        if pos is None:
            pos = self.position_ids[:, :seq_length]
        if typ is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(s[0], seq_length)
                typ = buffered_token_type_ids_expanded
            else:
                typ = torch.zeros(s, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.tok(x)
        token_type_embeddings = self.typ(typ)
        embeddings = inputs_embeds + token_type_embeddings
        if self.pos_type == "absolute":
            position_embeddings = self.pos(pos)
            embeddings += position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class YosoSelfAttention(qc.Module):
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
        self.pos_type = pos_type if pos_type is not None else config.pos_type

        self.use_expectation = config.use_expectation
        self.hash_code_len = config.hash_code_len
        self.use_conv = config.conv_window is not None
        self.use_fast_hash = config.use_fast_hash
        self.num_hash = config.num_hash
        self.lsh_backward = config.lsh_backward

        self.lsh_config = {
            "hash_code_len": self.hash_code_len,
            "use_fast_hash": self.use_fast_hash,
            "num_hash": self.num_hash,
            "lsh_backward": self.lsh_backward,
        }

        if config.conv_window is not None:
            self.conv = nn.Conv2d(
                in_channels=config.n_heads,
                out_channels=config.n_heads,
                kernel_size=(config.conv_window, 1),
                padding=(config.conv_window // 2, 0),
                bias=False,
                groups=config.n_heads,
            )

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.n_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def forward(self, hiddens, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hiddens)

        key_layer = self.transpose_for_scores(self.key(hiddens))
        value_layer = self.transpose_for_scores(self.value(hiddens))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.use_conv:
            conv_value_layer = self.conv(value_layer * attention_mask[:, None, :, None])

        batch_size, n_heads, seq_len, head_dim = query_layer.size()

        query_layer = query_layer.reshape(batch_size * n_heads, seq_len, head_dim)
        key_layer = key_layer.reshape(batch_size * n_heads, seq_len, head_dim)
        value_layer = value_layer.reshape(batch_size * n_heads, seq_len, head_dim)

        # revert changes made by get_extended_attention_mask
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = (
            attention_mask.squeeze()
            .repeat(1, n_heads, 1)
            .reshape(batch_size * n_heads, seq_len)
            .int()
        )

        # The CUDA kernels are most efficient with inputs whose size is a multiple of a GPU's warp size (32). Inputs
        # smaller than this are padded with zeros.
        gpu_warp_size = 32

        if (not self.use_expectation) and head_dim < gpu_warp_size:
            pad_size = batch_size * n_heads, seq_len, gpu_warp_size - head_dim

            query_layer = torch.cat(
                [
                    query_layer,
                    torch.zeros(pad_size, device=query_layer.device),
                ],
                dim=-1,
            )
            key_layer = torch.cat(
                [
                    key_layer,
                    torch.zeros(pad_size, device=key_layer.device),
                ],
                dim=-1,
            )
            value_layer = torch.cat(
                [
                    value_layer,
                    torch.zeros(pad_size, device=value_layer.device),
                ],
                dim=-1,
            )

        if self.use_expectation or self.training:
            query_layer, key_layer = normalize([query_layer, key_layer])

        if self.use_expectation:
            context_layer = YosoCumulation.apply(
                attention_mask, attention_mask, query_layer, key_layer, value_layer, self.lsh_config
            )
        else:
            context_layer = YosoLSHCumulation.apply(
                attention_mask, attention_mask, query_layer, key_layer, value_layer, self.lsh_config
            )

        if (not self.use_expectation) and head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :head_dim]

        context_layer = normalize(context_layer)

        context_layer = context_layer.reshape(batch_size, n_heads, seq_len, head_dim)

        if self.use_conv:
            context_layer += conv_value_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, context_layer) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class YosoSelfOutput(qc.Module):
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
        self.self = YosoSelfAttention(config, pos_type=pos_type)
        self.output = YosoSelfOutput(config)

    def forward(self, hiddens, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hiddens, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hiddens)
        outputs = (attention_output,) + self_outputs[1:]  # add attns if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class YosoIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


# Copied from transformers.models.bert.modeling_bert.BertOutput
class YosoOutput(qc.Module):
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
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = YosoIntermediate(config)
        self.output = YosoOutput(config)

    def forward(self, hiddens, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            hiddens, attention_mask, output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attns if we output attention weights

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
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hiddens,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(hiddens, attention_mask, output_attentions)

            hiddens = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        if not return_dict:
            return tuple(
                v for v in [hiddens, all_hidden_states, all_self_attentions] if v is not None
            )
        return qo.BaseWithCrossAttentions(
            y=hiddens,
            hiddens=all_hidden_states,
            attns=all_self_attentions,
        )


class Model(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = YosoEmbeddings(config)
        self.encoder = Encoder(config)

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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return qo.BaseWithCrossAttentions(
            y=sequence_output,
            hiddens=encoder_outputs.hiddens,
            attns=encoder_outputs.attns,
            crosses=encoder_outputs.crosses,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(cfg.d_model, **kw)

    forward = qf.forward_masked


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.yoso = Model(config)
        self.pre_classifier = qc.Linear(config.d_model, config.d_model)
        self.classifier = qc.Linear(config.d_model, 1)

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

        outputs = self.yoso(
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

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
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
        kw.update(n_labels=2)
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
