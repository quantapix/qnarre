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
        {"drop_proj"},
        dict(
            act="gelu_new",
            attn_type="block_sparse",
            block_size=64,
            BOS=1,
            d_ff=3072,
            d_model=768,
            drop_attn=0.1,
            drop=0.1,
            EOS=2,
            grad_checkpoint=True,
            init_range=0.02,
            is_enc_dec=False,
            eps=1e-12,
            model_type="big_bird",
            n_heads=12,
            n_lays=12,
            n_pos=4096,
            n_rand_blocks=3,
            n_typ=2,
            PAD=0,
            pos_type="absolute",
            rescale=False,
            s_vocab=50358,
            SEP=66,
            use_bias=True,
            y_cache=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
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

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, BigBirdEncoder):
            module.grad_checkpoint = value


MAP = {
    "google/bigbird-roberta-base": dict(
        archs=["ForPreTraining"],
        grad_checkpoint=False,
    ),
    "google/bigbird-roberta-large": dict(
        archs=["ForMasked"],
        d_ff=4096,
        d_model=1024,
        grad_checkpoint=False,
        n_heads=16,
        n_lays=24,
    ),
    "google/bigbird-base-trivia-itc": dict(
        archs=["ForQA"],
        grad_checkpoint=False,
        n_typ=16,
    ),
}
