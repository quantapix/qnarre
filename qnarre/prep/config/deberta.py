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
            act="gelu",
            d_ff=3072,
            d_hidden=768,
            drop_attn=0.1,
            drop=0.1,
            eps=1e-7,
            init_range=0.02,
            max_relative_positions=-1,
            model_type="deberta",
            n_heads=12,
            n_lays=12,
            n_pos=512,
            n_typ=0,
            PAD=0,
            pooler_dropout=0,
            pooler_hidden_act="gelu",
            pos_att_type=None,
            position_biased_input=True,
            relative_attention=False,
            s_vocab=50265,
            grad_checkpoint=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DebertaEncoder):
            module.gradient_checkpointing = value


MAP = {
    "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
    "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
    "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
    "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
    "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/config.json",
}
