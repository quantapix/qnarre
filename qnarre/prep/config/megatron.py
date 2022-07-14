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
            d_ff=4096,
            d_hidden=1024,
            drop_attn=0.1,
            drop=0.1,
            eps=1e-12,
            init_range=0.02,
            model_type="megatron-bert",
            n_heads=16,
            n_lays=24,
            n_pos=512,
            n_typ=2,
            PAD=0,
            pos_type="absolute",
            s_vocab=29056,
            y_cache=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, qc.Embed)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, qc.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MegatronBertEncoder):
            module.gradient_checkpointing = value


MAP = {}
