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

import json

from tokenizers import normalizers

from ....tokens.fast import PreTrainedTokenizerFast
from ..splinter import Tokenizer as Splinter

VOCAB_FS = {"vocab_file": "vocab.txt"}

VOCAB_MAP = {
    "vocab_file": {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/vocab.txt",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/vocab.txt",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/vocab.txt",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/vocab.txt",
    }
}

INPUT_CAPS = {
    "tau/splinter-base": 512,
    "tau/splinter-base-qass": 512,
    "tau/splinter-large": 512,
    "tau/splinter-large-qass": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "tau/splinter-base": {"do_lower_case": False},
    "tau/splinter-base-qass": {"do_lower_case": False},
    "tau/splinter-large": {"do_lower_case": False},
    "tau/splinter-large-qass": {"do_lower_case": False},
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    slow_tokenizer_class = Splinter

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk="[UNK]",
        sep="[SEP]",
        pad="[PAD]",
        cls="[CLS]",
        msk="[MASK]",
        question_token="[QUESTION]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kw,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=(question_token,),
            **kw,
        )
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)
        self.do_lower_case = do_lower_case

    @property
    def question_token_id(self):
        return self.convert_tokens_to_ids(self.question_token)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        suff = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            return cls + toks_0 + suff + sep + toks_1 + sep
        else:
            return cls + toks_0 + sep + toks_1 + suff + sep

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        suff = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        if self.padding_side == "right":
            return len(cls + toks_0 + suff + sep) * [0] + len(toks_1 + sep) * [1]
        else:
            return len(cls + toks_0 + sep) * [0] + len(toks_1 + suff + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))
