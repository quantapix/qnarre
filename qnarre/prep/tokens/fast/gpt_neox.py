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

VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

VOCAB_MAP = {
    "tokenizer_file": {
        "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt-neox-20b": 2048,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FS
    pretrained_vocab_files_map = VOCAB_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kw,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kw,
        )
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        self.add_prefix_space = add_prefix_space

    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def _build_conversation_input_ids(self, conversation):
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.EOS])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids
