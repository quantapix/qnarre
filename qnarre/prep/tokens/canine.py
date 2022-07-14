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


INPUT_CAPS = {
    "nielsr/canine-s": 2048,
}

UNICODE_VOCAB_SIZE = 1114112

PAD = 0

CLS = 0xE000
SEP = 0xE001
BOS = 0xE002
MASK = 0xE003
RESERVED = 0xE004

SPECIAL_CODEPOINTS = {
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

SPECIAL_CODEPOINTS_BY_NAME = {name: codepoint for codepoint, name in SPECIAL_CODEPOINTS.items()}


class Tokenizer(PreTrainedTokenizer):
    input_caps = INPUT_CAPS

    def __init__(
        self,
        bos=chr(CLS),
        eos=chr(SEP),
        sep=chr(SEP),
        cls=chr(CLS),
        pad=chr(PAD),
        msk=chr(MASK),
        add_prefix_space=False,
        model_max_length=2048,
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            bos=bos,
            eos=eos,
            sep=sep,
            cls=cls,
            pad=pad,
            msk=msk,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kw,
        )
        self._special_codepoints = {}
        for codepoint, name in SPECIAL_CODEPOINTS.items():
            self._special_codepoints[name] = codepoint
        self._special_codepoint_strings = {
            codepoint: name for name, codepoint in self._special_codepoints.items()
        }
        self._unicode_vocab_size = UNICODE_VOCAB_SIZE
        self._num_special_tokens = len(self._special_codepoints)

    @property
    def s_vocab(self):
        return self._unicode_vocab_size

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        try:
            return ord(token)
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index):
        try:
            if index in SPECIAL_CODEPOINTS:
                return SPECIAL_CODEPOINTS[index]
            return chr(index)
        except TypeError:
            raise ValueError(f"invalid id: {index}")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + toks_0 + sep
        if toks_1 is not None:
            result += toks_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)

        result = [1] + ([0] * len(toks_0)) + [1]
        if toks_1 is not None:
            result += ([0] * len(toks_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = len(cls + toks_0 + sep) * [0]
        if toks_1 is not None:
            result += len(toks_1 + sep) * [1]
        return result

    def save_vocabulary(self, dir, pre=None):
        return ()
