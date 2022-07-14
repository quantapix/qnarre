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

import copy
import torch

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            act_fun="relu",
            BOS=0,
            d_dec_ffn=4096,
            d_enc_ffn=4096,
            d_model=1024,
            dec_START=2,
            drop_act=0.0,
            drop_attn=0.0,
            drop_dec=0.0,
            drop_enc=0.0,
            drop=0.1,
            early_stop=False,
            EOS=2,
            forced_EOS=2,
            init_std=0.02,
            is_enc_dec=True,
            langs=["en", "de"],
            len_penalty=1.0,
            max_len=200,
            model_type="fsmt",
            n_beams=5,
            n_dec_heads=16,
            n_dec_lays=12,
            n_enc_heads=16,
            n_enc_lays=12,
            n_pos=1024,
            PAD=1,
            s_src_vocab=42024,
            s_tgt_vocab=42024,
            scale=True,
            tie_word_embeds=False,
            y_cache=True,
        ),
    )

    def _init_weights(self, module):
        std = self.cfg.init_std
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, qc.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad = self.cfg.PAD
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad]], device=self.device)
        dummy_inputs = {"mask": input_ids.ne(pad), "input_ids": input_ids}
        return dummy_inputs

    def to_dict(self):
        y = copy.deepcopy(self.__dict__)
        y["decoder"] = self.decoder.to_dict()
        y["model_type"] = self.__class__.model_type
        return y


MAP = {}
