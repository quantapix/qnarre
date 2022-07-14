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
from ....tokens.utils import is_sentencepiece_available


if is_sentencepiece_available():
    from ..fnet import Tokenizer as FNet
else:
    FNet = None

VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/tokenizer.json",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}

SPIECE_UNDERLINE = "▁"


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "token_type_ids"]
    slow_tokenizer_class = FNet

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        unk="<unk>",
        sep="[SEP]",
        pad="<pad>",
        cls="[CLS]",
        msk="[MASK]",
        **kw,
    ):
        msk = (
            AddedToken(msk, lstrip=True, rstrip=False, normalized=False)
            if isinstance(msk, str)
            else msk
        )
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            **kw,
        )
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return cls + toks_0 + sep
        return cls + toks_0 + sep + toks_1 + sep

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)
