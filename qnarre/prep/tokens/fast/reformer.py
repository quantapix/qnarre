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
from ....tokens.utils import is_sentencepiece_available


if is_sentencepiece_available():
    from ..reformer import Tokenizer as Reformer
else:
    Reformer = None


SPIECE_UNDERLINE = "▁"

VOCAB_FS = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "google/reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model"
    },
    "tokenizer_file": {
        "google/reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/tokenizer.json"
    },
}

INPUT_CAPS = {
    "google/reformer-crime-and-punishment": 524288,
}


class ReformerTokenizerFast(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Reformer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos="</s>",
        unk="<unk>",
        additional_special_tokens=[],
        **kw,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos=eos,
            unk=unk,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def save_vocabulary(self, dir, pre=None):
        assert self.can_save_slow_tokenizer
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)
