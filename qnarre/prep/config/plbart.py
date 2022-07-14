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
from collections import OrderedDict


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            act="gelu",
            BOS=0,
            d_model=768,
            decoder_attention_heads=12,
            decoder_ffn_dim=3072,
            decoder_layerdrop=0.0,
            decoder_layers=6,
            drop_act=0.0,
            drop_attn=0.1,
            drop_proj=0.0,
            drop=0.1,
            encoder_attention_heads=12,
            encoder_ffn_dim=3072,
            encoder_layerdrop=0.0,
            encoder_layers=6,
            EOS=2,
            forced_eos_token_id=2,
            init_std=0.02,
            is_enc_dec=True,
            model_type="plbart",
            n_pos=1024,
            PAD=1,
            s_vocab=50005,
            scale_embedding=True,
            y_cache=True,
            grad_checkpoint=True,
        ),
    )

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PLBartDecoder, PLBartEncoder)):
            module.gradient_checkpointing = value


MAP = {
    "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/config.json",
}


class PLBartOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self):
        if self.use_past:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),
                    ("past_keys", {0: "batch", 2: "sequence"}),
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),
                ]
            )
        else:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),
                ]
            )
