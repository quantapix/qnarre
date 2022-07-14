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

from ...tokens.utils import AddedToken, PreTrainedTokenizer


VOCAB_FS = {"vocab_file": "spiece.model"}

VOCAB_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model",
        "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model",
    }
}

INPUT_CAPS = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    prefix_tokens = []

    def __init__(
        self,
        vocab_file,
        unk="<unk>",
        bos="<s>",
        eos="</s>",
        pad="<pad>",
        sep="[SEP]",
        msk="[MASK]",
        cls="[CLS]",
        sp_model_kw=None,
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep

        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            bos=bos,
            eos=eos,
            unk=unk,
            pad=pad,
            sep=sep,
            msk=msk,
            cls=cls,
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
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def save_vocabulary(self, dir, pre=None):
        if not os.path.isdir(dir):
            logger.error(f"Vocabulary path ({dir}) should be a directory")
            return
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

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.SEP]
        cls = [self.cls_token_id]
        sep = [self.SEP]
        return cls + toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is None:
            return [1] + ([0] * len(toks_0)) + [1]
        return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]
