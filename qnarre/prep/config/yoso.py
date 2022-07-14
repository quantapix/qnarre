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
            BOS=0,
            conv_window=None,
            d_ff=3072,
            d_hidden=768,
            drop_attn=0.1,
            drop=0.1,
            EOS=2,
            eps=1e-12,
            hash_code_len=9,
            init_range=0.02,
            lsh_backward=True,
            model_type="yoso",
            n_heads=12,
            n_lays=12,
            n_pos=4096,
            n_typ=1,
            num_hash=64,
            PAD=1,
            pos_type="absolute",
            s_vocab=50265,
            use_expectation=True,
            use_fast_hash=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, YosoEncoder):
            module.gradient_checkpointing = value


MAP = {
    "uw-madison/yoso-4096": "https://huggingface.co/uw-madison/yoso-4096/resolve/main/config.json",
}
