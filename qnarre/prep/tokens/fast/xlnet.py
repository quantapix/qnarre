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
from .xlnet import Tokenizer as XLNet


VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/spiece.model",
        "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/tokenizer.json",
        "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "xlnet-base-cased": None,
    "xlnet-large-cased": None,
}

SPIECE_UNDERLINE = "▁"

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    padding_side = "left"
    slow_tokenizer_class = XLNet

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        sep="<sep>",
        pad="<pad>",
        cls="<cls>",
        msk="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kw,
    ):
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self._pad_token_type_id = 3
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return toks_0 + sep + cls
        return toks_0 + sep + toks_1 + sep + cls

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls_segment_id = [2]
        if toks_1 is None:
            return len(toks_0 + sep) * [0] + cls_segment_id
        return len(toks_0 + sep) * [0] + len(toks_1 + sep) * [1] + cls_segment_id

    def save_vocabulary(self, dir, pre=None):
        assert self.can_save_slow_tokenizer
        y = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(y):
            copyfile(self.vocab_file, y)
        return (y,)
