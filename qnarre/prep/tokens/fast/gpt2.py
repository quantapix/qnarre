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

from tokenizers import pre_tokenizers

from ....tokens.fast import PreTrainedTokenizerFast
from ..gpt2 import Tokenizer as GPT2


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

VOCAB_MAP = {
    "vocab_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/tokenizer.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/tokenizer.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/tokenizer.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    slow_tokenizer_class = GPT2

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk="<|endoftext|>",
        bos="<|endoftext|>",
        eos="<|endoftext|>",
        add_prefix_space=False,
        **kw,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk=unk,
            bos=bos,
            eos=eos,
            add_prefix_space=add_prefix_space,
            **kw,
        )
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kw):
        is_split_into_words = kw.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words
        return super()._batch_encode_plus(*args, **kw)

    def _encode_plus(self, *args, **kw):
        is_split_into_words = kw.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words
        return super()._encode_plus(*args, **kw)

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))

    def _build_conversation_input_ids(self, conversation):
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.EOS])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids
