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

import json
import os
import re
import unicodedata
import sacremoses as sm

from ...tokens.utils import PreTrainedTokenizer

VOCAB_FS = {
    "src_vocab_file": "vocab-src.json",
    "tgt_vocab_file": "vocab-tgt.json",
    "merges_file": "merges.txt",
}

VOCAB_MAP = {
    "src_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-src.json"
    },
    "tgt_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-tgt.json"
    },
    "merges_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/merges.txt"
    },
}

INPUT_CAPS = {"stas/tiny-wmt19-en-de": 1024}
PRETRAINED_INIT_CONFIGURATION = {
    "stas/tiny-wmt19-en-de": {
        "langs": ["en", "de"],
        "model_max_length": 1024,
        "special_tokens_map_file": None,
        "full_tokenizer_file": None,
    }
}


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def replace_unicode_punct(text):
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


def remove_non_printing_char(text):
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            continue
        output.append(char)
    return "".join(output)


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        langs=None,
        src_vocab_file=None,
        tgt_vocab_file=None,
        merges_file=None,
        do_lower_case=False,
        unk="<unk>",
        bos="<s>",
        sep="</s>",
        pad="<pad>",
        **kw,
    ):
        super().__init__(
            langs=langs,
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            unk=unk,
            bos=bos,
            sep=sep,
            pad=pad,
            **kw,
        )
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file
        self.merges_file = merges_file
        self.do_lower_case = do_lower_case
        self.cache_moses_punct_normalizer = dict()
        self.cache_moses_tokenizer = dict()
        self.cache_moses_detokenizer = dict()
        if langs and len(langs) == 2:
            self.src_lang, self.tgt_lang = langs
        else:
            raise ValueError(
                f"arg `langs` needs to be a list of 2 langs, e.g. ['en', 'ru'], but got {langs}. "
                "Usually that means that tokenizer can't find a mapping for the given model path "
                "in VOCAB_MAP, and other maps of this tokenizer."
            )
        with open(src_vocab_file, encoding="utf-8") as src_vocab_handle:
            self.encoder = json.load(src_vocab_handle)
        with open(tgt_vocab_file, encoding="utf-8") as tgt_vocab_handle:
            tgt_vocab = json.load(tgt_vocab_handle)
            self.decoder = {v: k for k, v in tgt_vocab.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def get_vocab(self):
        return self.get_src_vocab()

    @property
    def s_vocab(self):
        return self.s_src_vocab

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        return self.cache_moses_punct_normalizer[lang].normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    def moses_detokenize(self, tokens, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_detokenizer = sm.MosesDetokenizer(lang=self.tgt_lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        return self.cache_moses_detokenizer[lang].detokenize(tokens)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    @property
    def s_src_vocab(self):
        return len(self.encoder)

    @property
    def s_tgt_vocab(self):
        return len(self.decoder)

    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def get_tgt_vocab(self):
        return dict(self.decoder, **self.added_tokens_decoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text, lang="en", bypass_tokenizer=False):
        lang = self.src_lang
        if self.do_lower_case:
            text = text.lower()
        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)
        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk)

    def convert_tokens_to_string(self, tokens):
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        text = self.moses_detokenize(tokens, self.tgt_lang)
        return text

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        sep = [self.SEP]
        if toks_1 is None:
            return toks_0 + sep
        return toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is not None:
            return ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]
        return ([0] * len(toks_0)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        if toks_1 is None:
            return len(toks_0 + sep) * [0]
        return len(toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        src_vocab_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["src_vocab_file"],
        )
        tgt_vocab_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["tgt_vocab_file"],
        )
        merges_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["merges_file"],
        )
        with open(src_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))
        with open(tgt_vocab_file, "w", encoding="utf-8") as f:
            tgt_vocab = {v: k for k, v in self.decoder.items()}
            f.write(json.dumps(tgt_vocab, ensure_ascii=False))
        index = 0
        with open(merges_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merges_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1
        return src_vocab_file, tgt_vocab_file, merges_file
