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

from ....tokens.fast import PreTrainedTokenizerFast
from ..gpt import Tokenizer as GPT


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

VOCAB_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
    "tokenizer_file": {
        "openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/tokenizer.json"
    },
}

INPUT_CAPS = {
    "openai-gpt": 512,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    slow_tokenizer_class = GPT

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk="<unk>", **kw):
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk=unk, **kw)

    @property
    def do_lower_case(self):
        return True

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))
