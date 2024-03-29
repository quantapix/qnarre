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

import collections
import os
import unicodedata

from ...tokens.utils import (
    AddedToken,
    PreTrainedTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)

VOCAB_FS = {"vocab_file": "vocab.txt"}

VOCAB_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",
    }
}

INPUT_CAPS = {
    "microsoft/mpnet-base": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},
}


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        bos="<s>",
        eos="</s>",
        sep="</s>",
        cls="<s>",
        unk="[UNK]",
        pad="<pad>",
        msk="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            cls=cls,
            pad=pad,
            msk=msk,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kw,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk=self.unk)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def s_vocab(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + toks_0 + sep + sep + toks_1 + sep

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
        return [1] + ([0] * len(toks_0)) + [1, 1] + ([0] * len(toks_1)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep + sep + toks_1 + sep) * [0]

    def save_vocabulary(self, dir, pre=None):
        index = 0
        if os.path.isdir(dir):
            vocab_file = os.path.join(
                dir,
                (pre + "-" if pre else "") + VOCAB_FS["vocab_file"],
            )
        else:
            vocab_file = (pre + "-" if pre else "") + dir
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):
    def __init__(
        self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk = unk
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
