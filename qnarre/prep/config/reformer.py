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

import torch

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            act="relu",
            attention_head_size=64,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds_dim=[64, 192],
            axial_pos_embds=True,
            axial_pos_shape=[64, 64],
            chunk_size_lm_head=0,
            d_hidden=256,
            drop_proj=None,
            drop=0.05,
            EOS=2,
            eps=1e-12,
            feed_forward_size=512,
            hash_seed=None,
            init_range=0.02,
            is_decoder=False,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            local_num_chunks_after=0,
            local_num_chunks_before=1,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_attn_chunk_length=64,
            lsh_num_chunks_after=0,
            lsh_num_chunks_before=1,
            model_type="reformer",
            n_heads=12,
            n_pos=4096,
            num_buckets=None,
            num_hashes=1,
            PAD=0,
            s_vocab=320,
            tie_word_embeddings=False,
            y_cache=True,
        ),
    )

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        if isinstance(module, AxialPositionEmbeddings):
            for weight in module.weights:
                nn.init.normal_(weight, std=self.config.axial_norm_std)
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MAP = {
    "google/reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json",
    "google/reformer-enwik8": "https://huggingface.co/google/reformer-enwik8/resolve/main/config.json",
}
