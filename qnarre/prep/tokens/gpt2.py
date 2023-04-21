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
import regex as re

from functools import lru_cache

from ...tokens.utils import AddedToken, PreTrainedTokenizer


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

VOCAB_MAP = {
    "vocab_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/merges.txt",
    },
}

INPUT_CAPS = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk="<|endoftext|>",
        bos="<|endoftext|>",
        eos="<|endoftext|>",
        add_prefix_space=False,
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        super().__init__(
            errors=errors,
            unk=unk,
            bos=bos,
            eos=eos,
            add_prefix_space=add_prefix_space,
            **kw,
        )
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @property
    def s_vocab(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
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
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.bos:
            BOSs = [self.BOS]
        else:
            BOSs = []
        y = BOSs + token_ids_0
        if token_ids_1 is None:
            return y
        return y + BOSs + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0,
        token_ids_1=None,
        already_has_special_tokens=False,
    ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.bos:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        ys = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ys.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return ys

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    def convert_tokens_to_string(self, xs):
        y = "".join(xs)
        y = bytearray([self.byte_decoder[c] for c in y]).decode("utf-8", errors=self.errors)
        return y

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

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kw):
        add_prefix_space = kw.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kw)

    def _build_conversation_input_ids(self, conversation):
        ys = []
        for is_user, text in conversation.iter_texts():
            ys.extend(self.encode(text, add_special_tokens=False) + [self.EOS])
        if len(ys) > self.model_max_length:
            ys = ys[-self.model_max_length :]
        return ys
