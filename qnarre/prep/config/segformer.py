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
            classifier_dropout_prob=0.1,
            decoder_hidden_size=256,
            depths=[2, 2, 2, 2],
            drop_attn=0.0,
            drop_path_rate=0.1,
            drop=0.0,
            eps=1e-6,
            hidden_sizes=[32, 64, 160, 256],
            init_range=0.02,
            is_enc_dec=False,
            mlp_ratios=[4, 4, 4, 4],
            model_type="segformer",
            n_heads=[1, 2, 5, 8],
            num_channels=3,
            num_encoder_blocks=4,
            patch_sizes=[7, 3, 3, 3],
            semantic_loss_ignore_index=255,
            sr_ratios=[8, 4, 2, 1],
            strides=[4, 2, 2, 2],
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, nn.Conv2d)):
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


MAP = {
    "nvidia/segformer-b0-finetuned-ade-512-512": "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/config.json",
}
