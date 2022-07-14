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
            add_cross_attention=True,
            BOS=1,
            d_hidden=1024,
            decoder_ffn_dim=4096,
            decoder_start_token_id=0,
            disable_ngram_loss=False,
            drop_act=0.1,
            drop_attn=0.1,
            drop=0.1,
            encoder_ffn_dim=4096,
            EOS=2,
            eps=0.0,
            init_std=0.02,
            is_enc_dec=True,
            model_type="prophetnet",
            n_dec_lays=12,
            n_pos=512,
            ngram=2,
            num_buckets=32,
            num_decoder_attention_heads=16,
            num_encoder_attention_heads=16,
            num_encoder_layers=12,
            PAD=0,
            relative_max_distance=128,
            s_vocab=30522,
            y_cache=True,
            grad_checkpoint=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ProphetNetDecoder, ProphetNetEncoder)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        PAD = self.config.PAD
        assert decoder_start_token_id is not None
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        assert PAD is not None, "self.model.config.PAD has to be defined."
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, PAD)
        assert torch.all(shifted_input_ids >= 0).item()
        return shifted_input_ids


MAP = {
    "microsoft/prophetnet-large-uncased": "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json",
}
