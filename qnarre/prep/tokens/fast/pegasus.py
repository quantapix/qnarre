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

from ....tokens.fast import PreTrainedTokenizerFast
from ..pegasus import Tokenizer as Pegasus


SPIECE_UNDERLINE = "▁"

VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"
    },
    "tokenizer_file": {
        "google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/tokenizer.json"
    },
}

INPUT_CAPS = {
    "google/pegasus-xsum": 512,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    slow_tokenizer_class = Pegasus
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        msk="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,
        **kw,
    ):
        self.offset = offset
        if additional_special_tokens is not None:
            assert isinstance(additional_special_tokens, list)
            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )
            additional_special_tokens_extended += [
                f"<unk_{i}>"
                for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]
            if len(set(additional_special_tokens_extended)) != len(
                additional_special_tokens_extended
            ):
                raise ValueError(
                    f"Please make sure that the provided additional_special_tokens do not contain an incorrectly shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad=pad,
            eos=eos,
            unk=unk,
            msk=msk,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)
        all_special_ids.remove(self.unk_token_id)
        assert all_special_ids == set(range(len(self.additional_special_tokens) + 3))
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return self._special_token_mask(toks_0)
        elif toks_1 is None:
            return self._special_token_mask(toks_0) + [1]
        else:
            return self._special_token_mask(toks_0 + toks_1) + [1]

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return toks_0 + [self.EOS]
        return toks_0 + toks_1 + [self.EOS]

    def save_vocabulary(self, dir, pre=None):
        assert self.can_save_slow_tokenizer
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)
