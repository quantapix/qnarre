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
        [],
        dict(
            act_dim=4,
            act="relu",
            action_tanh=True,
            BOS=50256,
            d_hidden=128,
            drop_attn=0.1,
            drop_embed=0.1,
            drop_resid=0.1,
            EOS=50256,
            init_range=0.02,
            eps=1e-5,
            max_ep_len=4096,
            model_type="decision_transformer",
            n_embd=768,
            n_heads=1,
            n_inner=None,
            n_lays=3,
            n_pos=1024,
            reorder_and_upcast_attn=False,
            s_vocab=1,
            scale_attn_by_inverse_layer_idx=False,
            scale_attn_weights=True,
            state_dim=17,
            sum_act=None,
            drop_sum_first=0.1,
            sum_proj=True,
            sum_type="cls_index",
            sum_use_proj=True,
            y_cache=True,
            gradient_checkpoint=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                p.data.normal_(
                    mean=0.0,
                    std=(self.config.init_range / math.sqrt(2 * self.config.n_lays)),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DecisionTransformerGPT2Model):
            module.gradient_checkpointing = value


MAP = {
    "edbeeching/decision-transformer-gym-hopper-medium": "https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium/resolve/main/config.json",
}
