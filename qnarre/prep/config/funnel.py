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
            attention_type="relative_shift",
            block_repeats=None,
            block_sizes=[4, 4, 4],
            d_head=64,
            d_inner=3072,
            d_model=768,
            drop_act=0.0,
            drop_attn=0.1,
            drop=0.1,
            eps=1e-9,
            init_range=0.1,
            initializer_std=None,
            model_type="funnel",
            n_dec_lays=2,
            n_heads=12,
            n_pos=512,
            n_typ=3,
            pool_q_only=True,
            pooling_type="mean",
            s_vocab=30522,
            separate_cls=True,
            truncate_seq=True,
        ),
    )

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                qc.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                qc.init.constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            qc.init.uniform_(module.r_w_bias, b=self.config.init_range)
            qc.init.uniform_(module.r_r_bias, b=self.config.init_range)
            qc.init.uniform_(module.r_kernel, b=self.config.init_range)
            qc.init.uniform_(module.r_s_bias, b=self.config.init_range)
            qc.init.uniform_(module.seg_embed, b=self.config.init_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            qc.init.normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings.padding_idx is not None:
                module.word_embeddings.weight.data[module.padding_idx].zero_()


MAP = {
    "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/config.json",
    "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json",
    "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/config.json",
    "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/config.json",
    "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json",
    "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/config.json",
    "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/config.json",
    "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/config.json",
    "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json",
    "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/config.json",
}
