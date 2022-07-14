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
        {"drop_proj"},
        dict(
            act="gelu",
            act_sum="gelu",
            d_embed=128,
            d_ff=1024,
            d_model=256,
            drop_attn=0.1,
            drop_sum_last=0.1,
            drop=0.1,
            grad_checkpoint=True,
            init_range=0.02,
            model_type="electra",
            n_heads=4,
            n_lays=12,
            n_pos=512,
            n_typ=2,
            eps=1e-12,
            PAD=0,
            pos_type="absolute",
            s_vocab=30522,
            sum_type="first",
            sum_use_proj=True,
            y_cache=True,
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

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, ElectraEncoder):
            module.grad_checkpoint = value

    def __init__(self, **kw):
        super().__init__(PAD="PAD", **kw)


MAP = {
    "google/electra-small-generator": dict(
        archs=["ForMasked"],
    ),
    "google/electra-base-generator": dict(
        archs=["ForMasked"],
        d_embed=768,
    ),
    "google/electra-large-generator": dict(
        archs=["ForMasked"],
        d_embed=1024,
        n_lays=24,
    ),
    "google/electra-small-discriminator": dict(
        archs=["ForPreTraining"],
    ),
    "google/electra-base-discriminator": dict(
        archs=["ForPreTraining"],
        d_embed=768,
        d_ff=3072,
        d_model=768,
        n_heads=12,
    ),
    "google/electra-large-discriminator": dict(
        archs=["ForPreTraining"],
        d_embed=1024,
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
    ),
}


class Onnx:
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("mask", {0: "batch", 1: "sequence"}),
                ("typ_ids", {0: "batch", 1: "sequence"}),
            ]
        )
