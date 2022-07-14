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

import warnings

from ...tokens.utils import AddedToken, PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        extra_ids=125,
        additional_special_tokens=None,
        **kw,
    ):
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_tokens = len(
                set(filter(lambda x: ("extra_id" in str(x)), additional_special_tokens))
            )
            assert extra_tokens == extra_ids
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        super().__init__(
            eos=eos,
            unk=unk,
            pad=pad,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self._extra_ids = extra_ids
        self._utf_vocab_size = 2**8
        self.special_tokens_encoder = {
            self.pad: 0,
            self.eos: 1,
            self.unk: 2,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        n = len(additional_special_tokens)
        for i, token in enumerate(additional_special_tokens):
            self.special_tokens_encoder[token] = self.s_vocab + i - n
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens_encoder.items()}

    @property
    def s_vocab(self):
        return self._utf_vocab_size + self._num_special_tokens + self._extra_ids

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

    def _tokenize(self, text):
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index - self._num_special_tokens)
        return token

    def convert_tokens_to_string(self, tokens):
        bstring = b""
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    def save_vocabulary(self, dir, pre=None):
        return ()
