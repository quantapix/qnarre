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

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        {"n_inner", "act_sum"},
        dict(
            act="gelu_new",
            archs=["LMHead"],
            BOS=50256,
            drop_attn=0.1,
            drop_embed=0.1,
            drop_sum_first=0.1,
            drop=0.1,
            EOS=50256,
            eps=1e-5,
            init_range=0.02,
            model_type="gpt2",
            n_ctx=1024,
            n_heads=12,
            n_hidden=768,
            n_lays=12,
            n_pos=1024,
            reorder_and_upcast_attn=False,
            s_vocab=50257,
            scale_by_inv=False,
            scale=True,
            sum_proj=True,
            sum_type="cls_index",
            sum_use_proj=True,
            task_params={"text-generation": {"do_sample": True, "max_len": 50}},
            y_cache=True,
        ),
    )

    def __init__(self, *xs, **kw):
        super().__init__(*xs, **kw)

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
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                p.data.normal_(
                    mean=0.0,
                    std=(self.cfg.init_range / math.sqrt(2 * self.cfg.n_lays)),
                )

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, GPT2Model):
            module.grad_checkpoint = value


MAP = {
    "gpt2": dict(),
    "gpt2-medium": dict(
        n_heads=16,
        n_hidden=1024,
        n_lays=24,
        n_special=0,
        predict_special_tokens=True,
    ),
    "gpt2-large": dict(
        n_heads=20,
        n_hidden=1280,
        n_lays=36,
    ),
    "gpt2-xl": dict(
        n_heads=25,
        n_hidden=1600,
        n_lays=48,
        y_prev=True,
    ),
    "distilgpt2": dict(
        id2label={"0": "LABEL_0"},
        label2id={"LABEL_0": 0},
        n_labels=1,
        n_lays=6,
    ),
}
