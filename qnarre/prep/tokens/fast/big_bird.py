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

import os
from shutil import copyfile

from ....tokens.utils import AddedToken
from ....tokens.fast import PreTrainedTokenizerFast
from ..big_bird import Tokenizer as BigBird

VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model",
        "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/tokenizer.json",
        "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/tokenizer.json",
        "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}


SPIECE_UNDERLINE = "▁"


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    slow_tokenizer_class = BigBird
    model_input_names = ["input_ids", "mask"]
    prefix_tokens = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        unk="<unk>",
        bos="<s>",
        eos="</s>",
        pad="<pad>",
        sep="[SEP]",
        msk="[MASK]",
        cls="[CLS]",
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            **kw,
        )
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return cls + toks_0 + sep
        return cls + toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(self, toks_0, toks_1=None, has_specials=False):
        if has_specials:
            assert toks_1 is None
            return list(map(lambda x: 1 if x in [self.SEP, self.cls_token_id] else 0, toks_0))
        if toks_1 is None:
            return [1] + ([0] * len(toks_0)) + [1]
        return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        assert self.can_save_slow_tokenizer
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)
