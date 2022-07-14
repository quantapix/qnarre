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
            BOS=312,
            classifier_dropout_prob=0.1,
            d_ff=4608,
            d_hidden=1152,
            drop_attn=0.0,
            drop=0.0,
            EOS=313,
            eps=1e-12,
            init_range=0.02,
            input_embedding_size=256,
            is_enc_dec=False,
            model_type="rembert",
            n_heads=18,
            n_lays=32,
            n_pos=512,
            n_typ=2,
            output_embedding_size=1664,
            PAD=0,
            s_vocab=250300,
            y_cache=True,
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
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RemBertEncoder):
            module.gradient_checkpointing = value


MAP = {
    "rembert": "https://huggingface.co/google/rembert/resolve/main/config.json",
}
