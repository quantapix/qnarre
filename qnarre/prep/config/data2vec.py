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
            d_ff=3072,
            d_hidden=768,
            drop_attn=0.1,
            drop_proj=None,
            drop=0.1,
            EOS=2,
            eps=1e-12,
            init_range=0.02,
            model_type="data2vec-text",
            n_heads=12,
            n_lays=12,
            n_pos=512,
            n_typ=2,
            PAD=1,
            pos_type="absolute",
            s_vocab=30522,
            y_cache=True,
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
        elif isinstance(module, qc.LayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Data2VecTextEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        if not config.tie_word_embeddings:
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


MAP = {
    "facebook/data2vec-text-base": "https://huggingface.co/data2vec/resolve/main/config.json",
}
