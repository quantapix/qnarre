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
            attention_window=512,
            BOS=0,
            d_model=1024,
            decoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layerdrop=0.0,
            decoder_layers=12,
            decoder_start_token_id=2,
            drop_act=0.0,
            drop_attn=0.0,
            drop_proj=0.0,
            drop=0.1,
            encoder_attention_heads=16,
            encoder_ffn_dim=4096,
            encoder_layerdrop=0.0,
            encoder_layers=12,
            EOS=2,
            init_std=0.02,
            is_enc_dec=True,
            max_decoder_position_embeddings=1024,
            max_encoder_position_embeddings=16384,
            model_type="led",
            PAD=1,
            s_vocab=50265,
            y_cache=True,
            gradient_checkpoint=True,
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
        if isinstance(module, (LEDDecoder, LEDEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad = self.config.PAD
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad]], device=self.device)
        dummy_inputs = {"attention_mask": input_ids.ne(pad), "input_ids": input_ids}
        return dummy_inputs


MAP = {
    "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/config.json",
}
