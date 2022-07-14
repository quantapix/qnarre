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
from ..t5 import Tokenizer as T5


VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/tokenizer.json",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/tokenizer.json",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/tokenizer.json",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    slow_tokenizer_class = T5
    prefix_tokens = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kw,
    ):
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_tokens = len(
                set(filter(lambda x: ("extra_id_" in str(x)), additional_special_tokens))
            )
            assert extra_tokens == extra_ids
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos=eos,
            unk=unk,
            pad=pad,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True
        self._extra_ids = extra_ids

    def save_vocabulary(self, dir, pre=None):
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        toks_0 = toks_0 + [self.EOS]
        if toks_1 is None:
            return self.prefix_tokens + toks_0
        else:
            toks_1 = toks_1 + [self.EOS]
            return self.prefix_tokens + toks_0 + toks_1

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        eos = [self.EOS]
        if toks_1 is None:
            return len(toks_0 + eos) * [0]
        return len(toks_0 + eos + toks_1 + eos) * [0]
