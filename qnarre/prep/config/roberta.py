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

from collections import OrderedDict

from ... import core as qc
from .bert import PreTrained


class Roberta(PreTrained):
    hs = qc.Hypers(
        kw=dict(
            act="gelu",
            BOS=0,
            d_ff=3072,
            d_model=768,
            drop_attn=0.1,
            drop=0.1,
            EOS=2,
            eps=1e-05,
            grad_checkpoint=True,
            init_range=0.02,
            model_type="roberta",
            n_heads=12,
            n_lays=12,
            n_pos=514,
            n_typ=1,
            PAD=1,
            s_vocab=50265,
        )
    )

    def __init__(self, PAD=1, BOS=0, EOS=2, **kw):
        super().__init__(PAD=PAD, BOS=BOS, EOS=EOS, **kw)

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, Encoder):
            module.grad_checkpoint = value


MAP = {
    "roberta-base": dict(
        archs=["ForMasked"],
    ),
    "roberta-large": dict(
        archs=["ForMasked"],
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
    ),
    "roberta-large-mnli": dict(
        _num_labels=3,
        archs=["ForSeqClassifier"],
        d_ff=4096,
        d_model=1024,
        id2label={"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"},
        label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2},
        n_heads=16,
        n_lays=24,
    ),
    "distilroberta-base": dict(
        archs=["ForMasked"],
        n_lays=6,
    ),
    "roberta-base-openai-detector": dict(
        archs=["ForSeqClassifier"],
        y_prev=True,
    ),
    "roberta-large-openai-detector": dict(
        archs=["ForSeqClassifier"],
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
        y_prev=True,
    ),
}


class Onnx:
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("mask", {0: "batch", 1: "sequence"}),
            ]
        )
