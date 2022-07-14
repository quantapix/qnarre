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

from torch import nn

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        kw=dict(
            adaptive=True,
            clamp_len=1000,
            cutoffs=[20000, 40000, 200000],
            d_embed=1024,
            d_ff=4096,
            d_head=64,
            d_model=1024,
            div_val=4,
            drop_attn=0.0,
            drop=0.1,
            EOS=0,
            eps=1e-5,
            init_range=0.01,
            init_std=0.02,
            init="normal",
            mem_len=1600,
            model_type="transfo-xl",
            n_heads=16,
            n_lays=18,
            pre_lnorm=False,
            proj_init_std=0.01,
            proj_share_all_but_first=True,
            s_vocab=267735,
            same_length=True,
            sample_softmax=-1,
            untie_r=True,
        ),
    )

    def __init__(self, **kw):
        self.cutoffs = []
        self.cutoffs.extend(cutoffs)
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        super().__init__(EOS=EOS, **kw)

    def _init_weight(self, weight):
        if self.cfg.init == "uniform":
            nn.init.uniform_(weight, -self.cfg.init_range, self.cfg.init_range)
        elif self.cfg.init == "normal":
            nn.init.normal_(weight, 0.0, self.cfg.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.cfg.proj_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.cfg.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.cfg.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)


MAP = {
    "transfo-xl-wt103": dict(
        archs=["LMHead"],
        ext_len=0,
        task_params={"text-generation": {"do_sample": True, "max_len": 250}},
        tgt_len=128,
        tie_projs=[False, True, True, True],
        tie_weight=True,
    ),
}
