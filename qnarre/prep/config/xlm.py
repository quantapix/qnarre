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
        {"act_sum"},
        dict(
            asm=False,
            BOS=0,
            causal=False,
            d_model=2048,
            drop_attn=0.1,
            drop_sum_first=0.1,
            drop=0.1,
            embed_init_std=2048**-0.5,
            end_n_top=5,
            EOS=1,
            eps=1e-12,
            gelu_activation=True,
            init_std=0.02,
            is_enc=True,
            lang_embeds=True,
            LANG=0,
            model_type="xlm",
            MSK_TOK=0,
            MSK=5,
            n_heads=16,
            n_langs=1,
            n_lays=12,
            n_pos=512,
            PAD=2,
            s_vocab=30145,
            sin_embeds=False,
            start_n_top=5,
            sum_proj=True,
            sum_type="first",
            sum_use_proj=True,
            UNK=3,
        ),
    )

    @property
    def dummy_inputs(self):
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.cfg.lang_embeds and self.cfg.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "mask": attns_list, "langs": langs_list}

    def _init_weights(self, module):
        if isinstance(module, qc.Embedding):
            if self.cfg is not None and self.cfg.embed_init_std is not None:
                qc.init.normal_(module.weight, mean=0, std=self.cfg.embed_init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, qc.Linear):
            if self.cfg is not None and self.cfg.init_std is not None:
                qc.init.normal_(module.weight, mean=0, std=self.cfg.init_std)
                if module.bias is not None:
                    qc.init.constant_(module.bias, 0.0)
        if isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, **kw):
        if "s_vocab" in kw:
            self.s_vocab = kw["s_vocab"]
        super().__init__(PAD=PAD, BOS=BOS, **kw)


MAP = {
    "xlm-mlm-en-2048": dict(
        act_sum=None,
        archs=["LMHead"],
        embed_init_std=0.02209708691207961,
    ),
    "xlm-mlm-ende-1024": dict(
        act_sum=None,
        archs=["LMHead"],
        d_model=1024,
        id2lang={"0": "de", "1": "en"},
        lang2id={"de": 0, "en": 1},
        max_vocab=-1,
        min_count=0,
        n_heads=8,
        n_langs=2,
        n_lays=6,
        s_vocab=64699,
        same_enc_dec=True,
        share_inout_emb=True,
    ),
    "xlm-mlm-enro-1024": dict(
        act_sum=None,
        archs=["LMHead"],
        d_model=1024,
        id2lang={"0": "en", "1": "ro"},
        lang2id={"en": 0, "ro": 1},
        max_vocab=-1,
        min_count=0,
        n_heads=8,
        n_langs=2,
        n_lays=6,
        s_vocab=64592,
        same_enc_dec=True,
        share_inout_emb=True,
    ),
    "xlm-mlm-tlm-xnli15-1024": dict(
        act_sum=None,
        archs=["LMHead"],
        d_model=1024,
        id2lang={"2": "de", "4": "en"},
        lang2id={"de": 2, "en": 4},
        max_vocab=95000,
        min_count=0,
        n_heads=8,
        n_langs=15,
        s_vocab=95000,
        same_enc_dec=True,
        share_inout_emb=True,
    ),
    "xlm-mlm-xnli15-1024": dict(
        act_sum=None,
        archs=["LMHead"],
        d_model=1024,
        id2lang={"2": "de", "4": "en"},
        lang2id={"de": 2, "en": 4},
        max_vocab=95000,
        min_count=0,
        n_heads=8,
        n_langs=15,
        s_vocab=95000,
        same_enc_dec=True,
        share_inout_emb=True,
    ),
    "xlm-clm-ende-1024": dict(
        act_sum=None,
        archs=["LMHead"],
        d_model=1024,
        id2lang={"0": "de", "1": "en"},
        lang2id={"de": 0, "en": 1},
        max_vocab=-1,
        min_count=0,
        n_heads=8,
        n_langs=2,
        n_lays=6,
        s_vocab=64699,
        same_enc_dec=True,
        share_inout_emb=True,
    ),
}
