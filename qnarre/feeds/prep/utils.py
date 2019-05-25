# Copyright 2019 Quantapix Authors. All Rights Reserved.
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

import unicodedata

import pathlib as pth

from collections import abc
from functools import lru_cache

from qnarre.neura import tf
from qnarre.feeds.prep import records as R


class Words(abc.Mapping):
    def __init__(self):
        self.by_word = {}

    def __len__(self):
        return len(self.by_word)

    def __iter__(self):
        return iter(self.by_word)

    def __getitem__(self, w):
        try:
            w = self.by_word[w]
        except KeyError:
            self.by_word[w] = w
        return w


CLS = '[CLS]'
EOS = '[EOS]'
MSK = '[MASK]'
PAD = '[PAD]'
SEP = '[SEP]'
SOS = '[SOS]'
UNK = '[UNK]'


class Vocab(Words):
    fixed = False

    def __init__(self, ps, vocab=None):
        self.by_word = {}
        self.by_idx = []
        if vocab:
            self.load(vocab)
        ps.update(PAD=self.append(PAD),
                  UNK=self.append(UNK),
                  CLS=self.append(CLS),
                  SOS=self.append(SOS),
                  SEP=self.append(SEP),
                  EOS=self.append(EOS),
                  MSK=self.append(MSK))
        ps.update(vocab=self)
        self.max_used = -1

    def __getitem__(self, w):
        if isinstance(w, str):
            i = self.by_word[w]
            self.max_used = max(i, self.max_used)
        else:
            self.by_idx[w]

    def load(self, v):
        if isinstance(v, pth.Path):
            v = R.load(v, {
                'toks': tf.VarLenFeature(tf.string),
            })
        for i, w in enumerate(v):
            w = w.strip()
            assert w not in self.by_word
            self.by_word[w] = i
            self.by_idx.append(w)
        self.fixed = True

    def append(self, w):
        try:
            i = self.by_word[w]
        except KeyError:
            i = len(self.by_word)
            assert i == len(self.by_idx)
            self.by_word[w] = i
            self.by_idx.append(w)
            self.max_used = i
        return i

    def record(self):
        return R.example({
            'toks':
            R.bytes_list_feat(self.by_idx[:self.max_used + 1]),
        })


def normalize(txt):
    return unicodedata.normalize('NFD', txt)


def is_accent(c):
    return unicodedata.category(c) == 'Mn'


def is_punct(c):
    n = ord(c)
    if ((n >= 33 and n <= 47) or (n >= 58 and n <= 64) or (n >= 91 and n <= 96)
            or (n >= 123 and n <= 126)):
        return True
    return unicodedata.category(c).startswith('P')


def is_chinese(c):
    n = ord(c)
    return ((n >= 0x2B820 and n <= 0x2CEAF) or (n >= 0x2A700 and n <= 0x2B73F)
            or (n >= 0x3400 and n <= 0x4DBF) or (n >= 0x20000 and n <= 0x2A6DF)
            or (n >= 0xF900 and n <= 0xFAFF) or (n >= 0x2B740 and n <= 0x2B81F)
            or (n >= 0x4E00 and n <= 0x9FFF)
            or (n >= 0x2F800 and n <= 0x2FA1F))


@lru_cache()
def bytes_to_code():
    bc = {b: chr(b) for b in range(ord("!"), ord("~") + 1)}
    bc.update({b: chr(b) for b in range(ord("¡"), ord("¬") + 1)})
    bc.update({b: chr(b) for b in range(ord("®"), ord("ÿ") + 1)})
    i = 0
    for b in range(2**8):
        if b not in bc:
            bc[b] = chr(2**8 + i)
            i += 1
    return bc, {c: b for b, c in bc.items()}


class Embeds:

    _word = _chars = None

    @property
    def word(self):
        return self._word

    @word.setter
    def word(self, v):
        self._word = tuple(v)

    @property
    def chars(self):
        return self._chars

    @chars.setter
    def chars(self, v):
        self._chars = tuple(v)


class Tokenizer:

    limit = 500

    def __init__(self, words=None, limit=None):
        super().__init__()
        self.words = words or Words()
        if limit is not None:
            self.limit = limit

    def __call__(self, topics):
        pass


class Embedder:

    words = chars = {}

    def __call__(self, topics):
        for _, c in topics.contexts:
            for t in c.tokens:
                self._do_token(t)
        for _, _, q in topics.questions:
            for t in q.tokens:
                self._do_token(t)

    def _do_token(self, token):
        w = token.word
        token.embeds.word = self.words.get(w, ())
        token.embeds.chars = (self.chars.get(c, ()) for c in w)


class Featurer:

    poss = ners = set()

    def __call__(self, topics):
        self.poss = set()
        self.ners = set()
        for _, c in topics.contexts:
            self.poss |= c.tokens.poss
            self.ners |= c.tokens.ners
        for _, _, q in topics.questions:
            self.poss |= q.tokens.poss
            self.ners |= q.tokens.ners


"""
from collections import abc, defaultdict

_uids = defaultdict(int)


def next_uid(key=None):
    _uids[key] += 1
    return _uids[key]
"""
