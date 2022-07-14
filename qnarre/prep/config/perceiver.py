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
            audio_samples_per_frame=1920,
            cross_attention_shape_for_attention="kv",
            cross_attention_widening_factor=1,
            d_latents=1280,
            d_model=768,
            drop_attn=0.1,
            eps=1e-12,
            image_size=56,
            init_range=0.02,
            is_enc_dec=False,
            model_type="perceiver",
            n_pos=2048,
            num_blocks=1,
            num_cross_attention_heads=8,
            num_frames=16,
            num_latents=256,
            num_self_attends_per_block=26,
            num_self_attention_heads=8,
            output_shape=[1, 16, 224, 224],
            position_embedding_init_scale=0.02,
            qk_channels=None,
            s_vocab=262,
            samples_per_patch=16,
            self_attention_widening_factor=1,
            train_size=[368, 496],
            use_query_residual=True,
            v_channels=None,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear, qc.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=self.config.init_range)
        elif hasattr(module, "position_embeddings") and isinstance(
            module, PerceiverTrainablePositionEncoding
        ):
            module.position_embeddings.data.normal_(mean=0.0, std=self.config.init_range)
        elif isinstance(module, nn.ParameterDict):
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=self.config.init_range)
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MAP = {
    "deepmind/language-perceiver": "https://huggingface.co/deepmind/language-perceiver/resolve/main/config.json",
}
