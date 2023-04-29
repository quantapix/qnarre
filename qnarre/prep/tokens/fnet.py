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
import unicodedata
import sentencepiece as spm

from shutil import copyfile

from ...tokens.utils import AddedToken, PreTrainedTokenizer

VOCAB_FS = {"vocab_file": "spiece.model"}

VOCAB_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
}

INPUT_CAPS = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}

SPIECE_UNDERLINE = "▁"


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        unk="<unk>",
        sep="[SEP]",
        pad="<pad>",
        cls="[CLS]",
        msk="[MASK]",
        sp_model_kw=None,
        **kw,
    ):
        msk = (
            AddedToken(msk, lstrip=True, rstrip=False, normalized=False)
            if isinstance(msk, str)
            else msk
        )
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(vocab_file)

    @property
    def s_vocab(self):
        return len(self.sp_model)

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

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()
        return outputs

    def _tokenize(self, text):
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return cls + toks_0 + sep
        return cls + toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)

        if toks_1 is not None:
            return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]
        return [1] + ([0] * len(toks_0)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path) and os.path.isfile(
            self.vocab_file
        ):
            copyfile(self.vocab_file, path)
        elif not os.path.isfile(self.vocab_file):
            with open(path, "wb") as f:
                content_spiece_model = self.sp_model.serialized_model_proto()
                f.write(content_spiece_model)
        return (path,)
