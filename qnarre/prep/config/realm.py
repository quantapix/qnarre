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
            act="gelu_new",
            BOS=0,
            d_ff=3072,
            d_hidden=768,
            drop_attn=0.1,
            drop=0.1,
            EOS=2,
            eps=1e-12,
            init_range=0.02,
            max_span_width=10,
            model_type="realm",
            n_heads=12,
            n_lays=12,
            n_pos=512,
            n_typ=2,
            num_block_records=13353718,
            num_candidates=8,
            PAD=1,
            reader_beam_size=5,
            reader_layer_norm_eps=1e-3,
            reader_seq_len=320,  # 288 + 32
            retriever_proj_size=128,
            s_vocab=30522,
            searcher_beam_size=5000,
            searcher_seq_len=64,
            span_hidden_size=256,
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

    def _flatten_inputs(self, *inputs):
        flattened_inputs = []
        for tensor in inputs:
            if tensor is None:
                flattened_inputs.append(None)
            else:
                input_shape = tensor.shape
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                flattened_inputs.append(tensor)
        return flattened_inputs


MAP = {
    "google/realm-cc-news-pretrained-embedder": "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/config.json",
    "google/realm-cc-news-pretrained-encoder": "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/config.json",
    "google/realm-cc-news-pretrained-scorer": "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/config.json",
    "google/realm-cc-news-pretrained-openqa": "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/config.json",
    "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/config.json",
    "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/config.json",
    "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/config.json",
    "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/config.json",
}
