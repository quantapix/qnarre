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

from ...tokens.utils import PreTrainedTokenizer
from .bert import Tokenizer as Bert


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

VOCAB_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
}

INPUT_CAPS = {
    "openai-gpt": 512,
}


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(
        r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text
    )
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]

    def __init__(self, vocab_file, merges_file, unk="<unk>", **kw):
        super().__init__(unk=unk, **kw)
        try:
            import ftfy
            from spacy.lang.en import English

            _nlp = English()
            self.nlp = _nlp.tokenizer
            self.fix_text = ftfy.fix_text
        except ImportError:
            logger.warning(
                "ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy."
            )
            self.nlp = Bert(do_lower_case=True)
            self.fix_text = None
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def do_lower_case(self):
        return True

    @property
    def s_vocab(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

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

    def _tokenize(self, text):
        split_tokens = []
        if self.fix_text is None:
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])
        else:
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend([t for t in self.bpe(token.text.lower()).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk)

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    def save_vocabulary(self, dir, pre=None):
        vocab_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["vocab_file"],
        )
        merge_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["merges_file"],
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1
        return vocab_file, merge_file
