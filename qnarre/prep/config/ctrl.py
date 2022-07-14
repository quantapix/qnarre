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

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        {"act_sum", "finetune"},
        dict(
            d_ff=8192,
            d_model=1280,
            drop_attn=0.1,
            drop_resid=0.1,
            drop_sum_first=0.1,
            drop=0.1,
            eps=1e-6,
            init_range=0.02,
            model_type="ctrl",
            n_ctx=512,
            n_heads=16,
            n_labels=1,
            n_lays=48,
            n_pos=256,
            s_vocab=246534,
            sum_proj=True,
            sum_type="cls_index",
            sum_use_proj=True,
            y_cache=True,
        ),
    )

    def _init_weights(self, m):
        if isinstance(m, (qc.Linear, qc.Conv1D)):
            m.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, qc.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, qc.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)


MAP = {
    "ctrl": dict(
        from_tf=False,
        n_pos=50000,
        eps=1e-06,
        torchscript=False,
        use_bfloat16=False,
    )
}
