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

from .. import core as qc
from ..core import utils as qu
from ..core import output as qo
from ..core import attention as qa
from ..core.embed import Embeds
from ..core.mlp import Classifier, MLP, Masked, Pool
from ..prep.config.dpr import PreTrained


log = logging.get_logger(__name__)

from dataclasses import dataclass

from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig


DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


@dataclass
class Output(qo.Output):
    logits_beg = None
    logits_end = None
    relevance_logits = None
    hiddens = None
    attns = None


class Encoder(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        self.bert_model = BertModel(config, add_pooling_layer=False)
        if self.bert_model.config.d_model <= 0:
            raise ValueError("Encoder d_model can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = qc.Linear(self.bert_model.config.d_model, config.projection_dim)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return qo.BaseWithPooling(
            y=sequence_output,
            pools=pooled_output,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )

    @property
    def embeddings_size(self):
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.d_model


class SpanPredictor(PreTrained):
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.qa_outputs = qc.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = qc.Linear(self.encoder.embeddings_size, 1)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        n_passages, sequence_length = (
            input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        )
        # feed encoder
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # compute logits
        logits = self.qa_outputs(sequence_output)
        logits_beg, logits_end = logits.split(1, dim=-1)
        logits_beg = logits_beg.squeeze(-1).contiguous()
        logits_end = logits_end.squeeze(-1).contiguous()
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # resize
        logits_beg = logits_beg.view(n_passages, sequence_length)
        logits_end = logits_end.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)

        if not return_dict:
            return (logits_beg, logits_end, relevance_logits) + outputs[2:]

        return Output(
            logits_beg=logits_beg,
            logits_end=logits_end,
            relevance_logits=relevance_logits,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class DPRPretrainedContextEncoder(PreTrained):
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "ctx_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


class DPRPretrainedQuestionEncoder(PreTrained):
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "question_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


class DPRPretrainedReader(PreTrained):
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "span_predictor"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


class DPRContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = Encoder(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
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
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.PAD)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return qo.WithPools(
            pools=outputs.pools,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.question_encoder = Encoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
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
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.PAD)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return qo.WithPools(
            pools=outputs.pools,
            hiddens=outputs.hiddens,
            attns=outputs.attns,
        )


class Reader(DPRPretrainedReader):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.span_predictor = SpanPredictor(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
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

        return self.span_predictor(
            input_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
