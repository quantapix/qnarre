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


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        kw=dict(
            activation="gelu",
            d_model=768,
            drop_attn=0.1,
            drop=0.1,
            drop_qa=0.1,
            drop_seq=0.2,
            init_range=0.02,
            model_type="distilbert",
            n_heads=12,
            n_lays=6,
            n_pos=512,
            eps=1e-12,
            PAD=0,
            pos_sin=False,
            s_vocab=30522,
        ),
    )

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


MAP = {
    "distilbert-base-uncased": dict(
        archs=["ForMasked"],
    ),
    "distilbert-base-uncased-distilled-squad": dict(
        archs=["ForQA"],
    ),
    "distilbert-base-cased": dict(
        s_vocab=28996,
        y_prev=True,
    ),
    "distilbert-base-cased-distilled-squad": dict(
        archs=["ForQA"],
        pos_sin=True,
        s_vocab=28996,
        y_prev=True,
    ),
    "distilbert-base-german-cased": dict(
        archs=["ForMasked"],
        pos_sin=True,
        s_vocab=31102,
        y_prev=True,
    ),
    "distilbert-base-multilingual-cased": dict(
        archs=["ForMasked"],
        s_vocab=119547,
        y_prev=True,
    ),
    "distilbert-base-uncased-finetuned-sst-2-english": dict(
        archs=["ForSeqClassifier"],
        finetune="sst-2",
        id2label={"0": "NEGATIVE", "1": "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
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
