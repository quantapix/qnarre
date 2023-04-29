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

import sentencepiece as spm

from ...tokens.utils import PreTrainedTokenizer


SPIECE_UNDERLINE = "▁"

VOCAB_FS = {"vocab_file": "spiece.model"}

VOCAB_MAP = {
    "vocab_file": {
        "google/reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model"
    }
}

INPUT_CAPS = {
    "google/reformer-crime-and-punishment": 524288,
}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos="</s>",
        unk="<unk>",
        additional_special_tokens=[],
        sp_model_kw=None,
        **kw,
    ):
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            eos=eos,
            unk=unk,
            additional_special_tokens=additional_special_tokens,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(vocab_file)

    @property
    def s_vocab(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.s_vocab)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kw"):
            self.sp_model_kw = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def save_vocabulary(self, dir, pre=None):
        path = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(path) and os.path.isfile(
            self.vocab_file
        ):
            copyfile(self.vocab_file, path)
        elif not os.path.isfile(self.vocab_file):
            with open(path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (path,)
