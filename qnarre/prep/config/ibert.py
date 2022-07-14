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
            drop=0.1,
            EOS=2,
            eps=1e-12,
            force_dequant="none",
            init_range=0.02,
            model_type="ibert",
            n_heads=12,
            n_lays=12,
            n_pos=512,
            n_typ=2,
            PAD=1,
            pos_type="absolute",
            quant_mode=False,
            s_vocab=30522,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, (QuantLinear, qc.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (QuantEmbedding, qc.Embed)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (IntLayerNorm, qc.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError("`resize_token_embeddings` is not supported for I-BERT.")


MAP = {
    "kssteven/ibert-roberta-base": "https://huggingface.co/kssteven/ibert-roberta-base/resolve/main/config.json",
    "kssteven/ibert-roberta-large": "https://huggingface.co/kssteven/ibert-roberta-large/resolve/main/config.json",
    "kssteven/ibert-roberta-large-mnli": "https://huggingface.co/kssteven/ibert-roberta-large-mnli/resolve/main/config.json",
}
