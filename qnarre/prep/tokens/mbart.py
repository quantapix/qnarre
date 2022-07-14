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

from contextlib import contextmanager
from shutil import copyfile

from ...tokens.utils import AddedToken, PreTrainedTokenizer

SPIECE_UNDERLINE = "▁"

VOCAB_FS = {"vocab_file": "sentencepiece.bpe.model"}

VOCAB_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model",
        "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model",
    }
}

INPUT_CAPS = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

# fmt: off
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
# fmt: on


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    input_caps = INPUT_CAPS
    vocab_map = VOCAB_MAP
    model_input_names = ["input_ids", "mask"]

    prefix_tokens = []
    suffix_tokens = []

    def __init__(
        self,
        vocab_file,
        bos="<s>",
        eos="</s>",
        sep="</s>",
        cls="<s>",
        unk="<unk>",
        pad="<pad>",
        msk="<mask>",
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        sp_model_kw=None,
        additional_special_tokens=None,
        **kw,
    ):
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            cls=cls,
            pad=pad,
            msk=msk,
            tokenizer_file=None,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = 1
        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset
            for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = (
            len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset
        )
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        self._additional_special_tokens = list(self.lang_code_to_id.keys())
        if additional_special_tokens is not None:
            self._additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in self._additional_special_tokens]
            )
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kw"):
            self.sp_model_kw = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kw)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def s_vocab(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1

    @property
    def src_lang(self):
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang):
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if toks_1 is None:
            return prefix_ones + ([0] * len(toks_0)) + suffix_ones
        return prefix_ones + ([0] * len(toks_0)) + ([0] * len(toks_1)) + suffix_ones

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return self.prefix_tokens + toks_0 + self.suffix_tokens
        return self.prefix_tokens + toks_0 + toks_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep + sep + toks_1 + sep) * [0]

    def _build_translation_inputs(
        self,
        raw_inputs,
        return_tensors,
        src_lang,
        tgt_lang,
        **extra_kw,
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(
            raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kw
        )
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_BOS"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.s_vocab)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
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

    def prepare_seq2seq_batch(
        self,
        src_texts,
        src_lang="en_XX",
        tgt_texts=None,
        tgt_lang="ro_RO",
        **kw,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kw)

    @contextmanager
    def as_target_tokenizer(self):
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang):
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.EOS, self.cur_lang_code]

    def set_tgt_lang_special_tokens(self, lang):
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.EOS, self.cur_lang_code]
