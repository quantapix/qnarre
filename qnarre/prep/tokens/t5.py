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
import re
import warnings
import sentencepiece as spm

from shutil import copyfile

from ...tokens.utils import PreTrainedTokenizer

VOCAB_FS = {"vocab_file": "spiece.model"}

VOCAB_MAP = {
    "vocab_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
    }
}

INPUT_CAPS = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        vocab_file,
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kw=None,
        **kw,
    ):
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_tokens = len(
                set(filter(lambda x: ("extra_id" in str(x)), additional_special_tokens))
            )
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            eos=eos,
            unk=unk,
            pad=pad,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(vocab_file)

    @property
    def s_vocab(self):
        return self.sp_model.get_piece_size() + self._extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.s_vocab)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is None:
            return ([0] * len(toks_0)) + [1]
        return ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]

    def _add_eos_if_not_present(self, token_ids):
        if len(token_ids) > 0 and token_ids[-1] == self.EOS:
            warnings.warn(
                f"This sequence already has {self.eos}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.EOS]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        eos = [self.EOS]
        if toks_1 is None:
            return len(toks_0 + eos) * [0]
        return len(toks_0 + eos + toks_1 + eos) * [0]

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        toks_0 = self._add_eos_if_not_present(toks_0)
        if toks_1 is None:
            return toks_0
        else:
            toks_1 = self._add_eos_if_not_present(toks_1)
            return toks_0 + toks_1

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
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.s_vocab - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = f"<extra_id_{self.s_vocab - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

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
