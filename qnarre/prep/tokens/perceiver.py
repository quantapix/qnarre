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

from ...tokens.utils import AddedToken, PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad="[PAD]",
        bos="[BOS]",
        eos="[EOS]",
        msk="[MASK]",
        cls="[CLS]",
        sep="[SEP]",
        model_max_length=2048,
        **kw,
    ):
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        msk = AddedToken(msk, lstrip=False, rstrip=False) if isinstance(msk, str) else msk
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep
        super().__init__(
            pad=pad,
            bos=bos,
            eos=eos,
            msk=msk,
            cls=cls,
            sep=sep,
            model_max_length=model_max_length,
            **kw,
        )
        self._utf_vocab_size = 2**8
        self.special_tokens_encoder = {
            self.pad: 0,
            self.bos: 1,
            self.eos: 2,
            self.msk: 3,
            self.cls: 4,
            self.sep: 5,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens_encoder.items()}

    def get_vocab(self):
        vocab = self.special_tokens_encoder.copy()
        vocab.update(self.added_tokens_encoder)
        for i in range(self._utf_vocab_size):
            token = chr(i)
            vocab[token] = i + len(self.special_tokens_encoder)
        return vocab

    @property
    def s_vocab(self):
        return self._utf_vocab_size + self._num_special_tokens

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is None:
            return [1] + [0] * len(toks_0) + [1]
        return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id]
        else:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id] + toks_1 + [self.sep_token_id]

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
        elif index in self.added_tokens_decoder:
            token = self.added_tokens_decoder[index]
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
        string = bstring.decode("utf-8", errors="replace")
        return string

    def save_vocabulary(self, dir, pre=None):
        return ()
