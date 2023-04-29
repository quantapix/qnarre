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
            d_ff=24576,
            d_hidden=6144,
            EOS=2,
            eps=1e-5,
            init_range=0.02,
            model_type="gpt_neox",
            n_heads=64,
            n_lays=44,
            n_pos=2048,
            rotary_emb_base=10000,
            rotary_pct=0.25,
            s_vocab=50432,
            tie_word_embeddings=False,
            use_parallel_residual=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoXModel):
            module.gradient_checkpointing = value


MAP = {
    "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json",
}
