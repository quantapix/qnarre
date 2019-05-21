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

from collections import abc, defaultdict


def normalize(txt):
    return ' '.join(unicodedata.normalize('NFD', txt).split())


_uids = defaultdict(int)


def next_uid(key=None):
    _uids[key] += 1
    return _uids[key]


class Words(abc.Mapping):
    def __init__(self):
        self.by_word = {'': ''}

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


class Indeces(Words):
    def __init__(self):
        self.by_word = {None: 0, '': 1}
        self.by_idx = [None, '']

    def __getitem__(self, w):
        try:
            return self.by_word[w] if isinstance(w, str) else self.by_idx[w]
        except KeyError:
            return 0 if isinstance(w, str) else None

    def append(self, w):
        if w not in self.by_word:
            self.by_word[w] = len(self.by_word)
            self.by_idx.append(w)


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
