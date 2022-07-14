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
        [],
        dict(
            act="gelu_new",
            BOS=2,
            d_embed=128,
            d_ff=16384,
            d_model=4096,
            down_scale_factor=1,
            drop_attn=0,
            drop_proj=0.1,
            drop=0,
            EOS=3,
            init_range=0.02,
            model_type="albert",
            n_groups=1,
            n_heads=64,
            n_lays=12,
            n_mem_blocks=0,
            n_pos=512,
            n_typ=2,
            net_type=0,
            eps=1e-12,
            PAD=0,
            pos_type="absolute",
            s_gap=0,
            s_group=1,
            s_vocab=30000,
        ),
    )

    def _init_weights(self, m):
        if isinstance(m, qc.Linear):
            m.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, qc.Embed):
            m.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, qc.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)


MAP = {
    "albert-base-v1": dict(
        act="gelu",
        archs=["ForMasked"],
        d_ff=3072,
        d_model=768,
        drop_attn=0.1,
        drop=0.1,
        n_heads=12,
    ),
    "albert-large-v1": dict(
        act="gelu",
        archs=["ForMasked"],
        d_ff=4096,
        d_model=1024,
        drop_attn=0.1,
        drop=0.1,
        n_heads=16,
        n_lays=24,
    ),
    "albert-xlarge-v1": dict(
        act="gelu",
        archs=["ForMasked"],
        d_ff=8192,
        d_model=2048,
        drop_attn=0.1,
        drop=0.1,
        n_heads=16,
        n_lays=24,
    ),
    "albert-xxlarge-v1": dict(
        act="gelu",
        archs=["ForMasked"],
    ),
    "albert-base-v2": dict(
        archs=["ForMasked"],
        d_ff=3072,
        d_model=768,
        n_heads=12,
    ),
    "albert-large-v2": dict(
        archs=["ForMasked"],
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
    ),
    "albert-xlarge-v2": dict(
        archs=["ForMasked"],
        d_ff=8192,
        d_model=2048,
        n_heads=16,
        n_lays=24,
    ),
    "albert-xxlarge-v2": dict(
        archs=["ForMasked"],
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
