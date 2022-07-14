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

from functools import partial
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
from ..prep.config.fnet import PreTrained


log = logging.get_logger(__name__)


from torch.nn import CrossEntropyLoss

from ...utils import is_scipy_available


if is_scipy_available():
    from scipy import linalg

from ...pytorch_utils import apply_chunking_to_forward

LIST = ["google/fnet-base", "google/fnet-large"]


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = x.type(torch.complex64)
    return torch.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


# # Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def fftn(x):
    out = x
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        out = torch.fft.fft(out, axis=axis)
    return out


class FNetEmbeddings(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = qc.Embed(config.s_vocab, config.d_model, padding_idx=config.PAD)
        self.position_embeddings = qc.Embed(config.n_pos, config.d_model)
        self.token_type_embeddings = qc.Embed(config.n_typ, config.d_model)
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)
        self.projection = qc.Linear(config.d_model, config.d_model)
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
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class FNetBasicFourierTransform(qc.Module):
    def __init__(self, config):
        super().__init__()
        self._init_fourier_transform(config)

    def _init_fourier_transform(self, config):
        if not config.use_tpu_fourier_optimizations:
            self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))
        elif config.n_pos <= 4096:
            if is_scipy_available():
                self.register_buffer(
                    "dft_mat_hidden",
                    torch.tensor(linalg.dft(config.d_model), dtype=torch.complex64),
                )
                self.register_buffer(
                    "dft_mat_seq",
                    torch.tensor(linalg.dft(config.tpu_short_seq_length), dtype=torch.complex64),
                )
                self.fourier_transform = partial(
                    two_dim_matmul,
                    matrix_dim_one=self.dft_mat_seq,
                    matrix_dim_two=self.dft_mat_hidden,
                )
            else:
                self.fourier_transform = fftn
        else:
            self.fourier_transform = fftn

    def forward(self, hiddens):
        outputs = self.fourier_transform(hiddens).real
        return (outputs,)


class FNetBasicOutput(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = qc.LayerNorm(config.d_model, eps=config.eps)

    def forward(self, hiddens, input_tensor):
        hiddens = self.norm(input_tensor + hiddens)
        return hiddens


class FNetFourierTransform(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.self = FNetBasicFourierTransform(config)
        self.output = FNetBasicOutput(config)

    def forward(self, hiddens):
        self_outputs = self.self(hiddens)
        fourier_output = self.output(self_outputs[0], hiddens)
        outputs = (fourier_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->FNet
class FNetIntermediate(qc.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = qc.Linear(cfg.d_model, cfg.d_ff)
        self.act = qu.activation(cfg.act)

    def forward(self, x):
        y = self.dense(x)
        y = self.act(y)
        return y


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->FNet
class FNetOutput(qc.Module):
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


class FNetLayer(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1  # The dimension which has the sequence length
        self.fourier = FNetFourierTransform(config)
        self.intermediate = FNetIntermediate(config)
        self.output = FNetOutput(config)

    def forward(self, hiddens):
        self_fourier_outputs = self.fourier(hiddens)
        fourier_output = self_fourier_outputs[0]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, fourier_output
        )
        outputs = (layer_output,)
        return outputs

    def feed_forward_chunk(self, fourier_output):
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


class FNetEncoder(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FNetLayer(config) for _ in range(config.n_lays)])
        self.gradient_checkpointing = False

    def forward(self, hiddens, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hiddens,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hiddens
                )
            else:
                layer_outputs = layer_module(hiddens)

            hiddens = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hiddens,)

        if not return_dict:
            return tuple(v for v in [hiddens, all_hidden_states] if v is not None)

        return qo.Base(y=hiddens, hiddens=all_hidden_states)


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->FNet
class FNetPreTrainingHeads(qc.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = Masker(config)
        self.seq_relationship = qc.Linear(config.d_model, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class Model(PreTrained):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.pooler = Pool(config) if add_pooling_layer else None

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if (
            self.config.use_tpu_fourier_optimizations
            and seq_length <= 4096
            and self.config.tpu_short_seq_length != seq_length
        ):
            raise ValueError(
                "The `tpu_short_seq_length` in FNetConfig should be set equal to the sequence length being passed to the model when using TPU optimizations."
            )

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pools = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pools) + encoder_outputs[1:]

        return qo.BaseWithPooling(
            y=sequence_output,
            pools=pools,
            hiddens=encoder_outputs.hiddens,
        )


class FNetForPreTraining(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.fnet = Model(config)
        self.cls = FNetPreTrainingHeads(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.s_vocab), labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return qo.LossSeq(
            loss=total_loss,
            logits=prediction_scores,
            orders=seq_relationship_score,
            hiddens=outputs.hiddens,
        )


class ForMasked(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = Masker(**kw)

    forward = qf.forward_masked


class FNetForNextSentencePrediction(PreTrained):
    def __init__(self, config):
        super().__init__(config)

        self.fnet = Model(config)
        self.cls = qc.Linear(config.d_model, 2)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kw,
    ):
        if "next_sentence_label" in kw:
            labels = kw.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hiddens=outputs.hiddens,
        )


class ForMultiChoice(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.fnet = Model(config)
        self.drop = qc.Dropout(config.drop)
        self.classifier = qc.Linear(config.d_model, 1)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
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

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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

        return qo.WithLoss(loss=loss, logits=reshaped_logits, hiddens=outputs.hiddens)


class ForSeqClassifier(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
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


class ForQA(PreTrained):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = self.get_cfg(kw)
        self.model = Model(**kw)
        self.proj = qc.Linear(cfg.d_model, cfg.n_labels, **kw)

    forward = qf.forward_qa
