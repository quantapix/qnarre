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
        kw=dict(
            act_fun="gelu",
            d_dec_ffn=4096,
            d_enc_ffn=4096,
            d_model=1024,
            dec_START=0,
            drop_act=0.0,
            drop_attn=0.0,
            drop_dec=0.0,
            drop_enc=0.0,
            drop_proj=0.0,
            drop=0.1,
            EOS=1,
            forced_EOS=1,
            grad_checkpoint=True,
            init_std=0.02,
            is_enc_dec=True,
            model_type="pegasus",
            n_dec_heads=16,
            n_dec_lays=12,
            n_enc_heads=16,
            n_enc_lays=12,
            n_pos=1024,
            PAD=0,
            s_vocab=50265,
            scale=False,
            y_cache=True,
        ),
    )

    def __init__(self, **kw):
        super().__init__(**kw)

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

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, (Decoder, Encoder)):
            module.grad_checkpoint = value

    @property
    def n_heads(self):
        return self.n_enc_heads


MAP = {
    "google/pegasus-large": dict(
        act_fun="relu",
        add_bias_logits=False,
        add_final_norm=True,
        archs=["ForCondGen"],
        BOS=0,
        drop_act=0.1,
        drop_attn=0.1,
        extra_pos_embeddings=1,
        force_bos_token_to_be_generated=False,
        grad_checkpoint=False,
        id2label={"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},
        label2id={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        len_penalty=0.8,
        max_len=256,
        n_beams=8,
        n_dec_lays=16,
        n_enc_lays=16,
        n_lays=16,
        name_or_path="google/pegasus-large",
        normalize_embedding=False,
        pre_norm=True,
        s_vocab=96103,
        scale=True,
        static_position_embeddings=True,
        task_params=dict(
            sum_aeslc=dict(
                len_penalty=0.6,
                max_len=32,
                n_pos=512,
            ),
            sum_arxiv=dict(
                len_penalty=0.8,
                max_len=256,
                n_pos=1024,
            ),
            sum_big_patent=dict(
                len_penalty=0.7,
                max_len=256,
                n_pos=1024,
            ),
            sum_billsum=dict(
                len_penalty=0.6,
                max_len=256,
                n_pos=1024,
            ),
            sum_cnn_dailymail=dict(
                len_penalty=0.8,
                max_len=128,
                n_pos=1024,
            ),
            sum_gigaword=dict(
                len_penalty=0.6,
                max_len=32,
                n_pos=128,
            ),
            sum_large=dict(
                len_penalty=0.8,
                max_len=256,
                n_pos=1024,
            ),
            sum_multi_news=dict(
                len_penalty=0.8,
                max_len=256,
                n_pos=1024,
            ),
            sum_newsroom=dict(
                len_penalty=0.8,
                max_len=128,
                n_pos=512,
            ),
            sum_pubmed=dict(
                len_penalty=0.8,
                max_len=256,
                n_pos=1024,
            ),
            sum_reddit_tifu=dict(
                len_penalty=0.6,
                max_len=128,
                n_pos=512,
            ),
            sum_wikihow=dict(
                len_penalty=0.6,
                max_len=256,
                n_pos=512,
            ),
            sum_xsum=dict(
                len_penalty=0.8,
                max_len=64,
                n_pos=512,
            ),
        ),
    )
}
