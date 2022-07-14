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
import sentencepiece as spm

from shutil import copyfile

from ...tokens.utils import PreTrainedTokenizer


SPIECE_UNDERLINE = "▁"

VOCAB_FS = {"vocab_file": "spiece.model"}

VOCAB_MAP = {
    "vocab_file": {
        "google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"
    }
}

INPUT_CAPS = {
    "google/pegasus-xsum": 512,
}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        vocab_file,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        msk="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,
        sp_model_kw=None,
        **kw,
    ):
        self.offset = offset
        if additional_special_tokens is not None:
            assert isinstance(additional_special_tokens, list)
            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )
            additional_special_tokens_extended += [
                f"<unk_{i}>"
                for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]
            if len(set(additional_special_tokens_extended)) != len(
                additional_special_tokens_extended
            ):
                raise ValueError(
                    f"Please make sure that the provided additional_special_tokens do not contain an incorrectly shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            eos=eos,
            unk=unk,
            msk=msk,
            pad=pad,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        self.mask_token_sent = mask_token_sent
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(vocab_file)
        self.encoder[int, str] = {
            0: self.pad,
            1: self.eos,
        }
        if self.mask_token_sent is not None:
            self.encoder.update(
                {
                    2: self.mask_token_sent,
                    3: self.msk,
                }
            )
        if self.offset > 0:
            self.encoder.update(
                {i + 3: additional_special_tokens[i] for i in range(1, self.offset - 1)}
            )
        self.decoder[str, int] = {v: k for k, v in self.encoder.items()}

    @property
    def s_vocab(self):
        return len(self.sp_model) + self.offset

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
        if token in self.decoder:
            return self.decoder[token]
        elif token in self.added_tokens_decoder:
            return self.added_tokens_decoder[token]
        sp_id = self.sp_model.piece_to_id(token)
        return sp_id + self.offset

    def _convert_id_to_token(self, index):
        if index in self.encoder:
            return self.encoder[index]
        elif index in self.added_tokens_encoder:
            return self.added_tokens_encoder[index]
        else:
            token = self.sp_model.IdToPiece(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)
        all_special_ids.remove(self.unk_token_id)
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return self._special_token_mask(toks_0)
        elif toks_1 is None:
            return self._special_token_mask(toks_0) + [1]
        else:
            return self._special_token_mask(toks_0 + toks_1) + [1]

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return toks_0 + [self.EOS]
        return toks_0 + toks_1 + [self.EOS]

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
