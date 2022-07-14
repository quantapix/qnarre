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
            BOS=0xE000,
            d_ff=3072,
            d_hidden=768,
            downsampling_rate=4,
            drop_attn=0.1,
            drop=0.1,
            EOS=0xE001,
            eps=1e-12,
            init_range=0.02,
            is_enc_dec=False,
            local_transformer_stride=128,
            model_type="canine",
            n_heads=12,
            n_lays=12,
            n_pos=16384,
            n_typ=16,
            num_hash_buckets=16384,
            num_hash_functions=8,
            PAD=0,
            upsampling_kernel_size=4,
            y_cache=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, qc.Conv1d)):
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
        if isinstance(module, CanineEncoder):
            module.gradient_checkpointing = value


MAP = {
    "google/canine-s": "https://huggingface.co/google/canine-s/resolve/main/config.json",
}
