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

from contextlib import contextmanager
from shutil import copyfile
from tokenizers import processors

from ....tokens.utils import AddedToken
from ....tokens.fast import PreTrainedTokenizerFast
from ..mbart import Tokenizer as MBart


VOCAB_FS = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model",
        "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/tokenizer.json",
        "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

# fmt: off
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
# fmt: on


class MBartTokenizerFast(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    input_caps = INPUT_CAPS
    vocab_map = VOCAB_MAP
    model_input_names = ["input_ids", "mask"]
    slow_tokenizer_class = MBart

    prefix_tokens = []
    suffix_tokens = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos="<s>",
        eos="</s>",
        sep="</s>",
        cls="<s>",
        unk="<unk>",
        pad="<pad>",
        msk="<mask>",
        src_lang=None,
        tgt_lang=None,
        additional_special_tokens=None,
        **kw,
    ):
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos=bos,
            eos=eos,
            sep=sep,
            cls=cls,
            unk=unk,
            pad=pad,
            msk=msk,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True
        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()
        if additional_special_tokens is not None:
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )
        self.add_special_tokens({"additional_special_tokens": _additional_special_tokens})
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self):
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang):
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

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
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.EOS, self.cur_lang_code]
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(
                zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)
            ),
        )

    def set_tgt_lang_special_tokens(self, lang):
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.EOS, self.cur_lang_code]
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(
                zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)
            ),
        )

    def save_vocabulary(self, dir, pre=None):
        assert self.can_save_slow_tokenizer
        path = os.path.join(dir, (pre + "-" if pre else "") + VOCAB_FS["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(path):
            copyfile(self.vocab_file, path)
        return (path,)
