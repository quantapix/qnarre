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
        {"act_sum"},
        dict(
            act="gelu",
            drop_attn=0.1,
            drop_embed=0.1,
            drop_sum_first=0.1,
            drop=0.1,
            init_range=0.02,
            model_type="openai-gpt",
            n_ctx=512,
            n_embed=768,
            n_heads=12,
            n_lays=12,
            n_pos=512,
            eps=1e-5,
            predict_special_tokens=True,
            s_vocab=40478,
            sum_proj=True,
            sum_type="cls_index",
            sum_use_proj=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, qc.Conv1D)):
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
    "openai-gpt": dict(
        archs=["LMHead"],
        n_special=0,
        task_params={"text-generation": {"do_sample": True, "max_len": 50}},
    )
}
