# Copyright 2023 Quantapix Authors. All Rights Reserved.
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

from collections import OrderedDict

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        {"drop_proj"},
        dict(
            act="gelu",
            archs=["ForMasked"],
            d_ff=3072,
            d_model=768,
            drop_attn=0.1,
            drop=0.1,
            grad_checkpoint=True,
            init_range=0.02,
            model_type="bert",
            n_heads=12,
            n_lays=12,
            n_pos=512,
            n_typ=2,
            eps=1e-12,
            PAD=0,
            pos_type="absolute",
            s_vocab=30522,
            y_cache=True,
        ),
    )

    def _init_weights(self, x):
        cfg = self.cfg
        if isinstance(x, qc.Linear):
            x.weight.data.normal_(mean=0.0, std=cfg.init_range)
            if x.bias is not None:
                x.bias.data.zero_()
        elif isinstance(x, qc.Embed):
            x.weight.data.normal_(mean=0.0, std=cfg.init_range)
            if x.padding_idx is not None:
                x.weight.data[cfg.PAD].zero_()
        elif isinstance(x, qc.LayerNorm):
            x.bias.data.zero_()
            x.weight.data.fill_(1.0)

    def _set_grad_checkpoint(self, x, value=False):
        if isinstance(x, Encoder):
            x.grad_checkpoint = value


MAP = {
    "bert-base-uncased": dict(
        grad_checkpoint=False,
        n_lays=12,
    ),
    "bert-large-uncased": dict(
        d_ff=4096,
        d_model=1024,
        grad_checkpoint=False,
        n_heads=16,
        n_lays=24,
    ),
    "bert-base-cased": dict(
        grad_checkpoint=False,
        s_vocab=28996,
    ),
    "bert-large-cased": dict(
        d_ff=4096,
        d_model=1024,
        direction="bidi",
        grad_checkpoint=False,
        n_heads=16,
        n_lays=24,
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=28996,
    ),
    "bert-base-multilingual-uncased": dict(
        direction="bidi",
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=105879,
    ),
    "bert-base-multilingual-cased": dict(
        direction="bidi",
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=119547,
    ),
    "bert-base-chinese": dict(
        direction="bidi",
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=21128,
    ),
    "bert-base-german-cased": dict(
        s_vocab=30000,
    ),
    "bert-large-uncased-whole-word-masking": dict(
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
    ),
    "bert-large-cased-whole-word-masking": dict(
        d_ff=4096,
        d_model=1024,
        direction="bidi",
        n_heads=16,
        n_lays=24,
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=28996,
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": dict(
        archs=["ForQA"],
        d_ff=4096,
        d_model=1024,
        n_heads=16,
        n_lays=24,
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": dict(
        archs=["ForQA"],
        d_ff=4096,
        d_model=1024,
        direction="bidi",
        n_heads=16,
        n_lays=24,
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        s_vocab=28996,
    ),
    "bert-base-cased-finetuned-mrpc": dict(
        s_vocab=28996,
    ),
}


class Onnx:
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("mask", {0: "batch", 1: "sequence"}),
                ("typ_ids", {0: "batch", 1: "sequence"}),
            ]
        )
