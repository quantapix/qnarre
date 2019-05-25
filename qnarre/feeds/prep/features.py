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

import copy

import collections as col
import collections.abc as abc

Span = col.namedtuple('Span', 'beg end')
Token = col.namedtuple('Token', 'text span pos lemma ner embs')


class Topic(col.namedtuple('Topic', 'text contexts uid toks')):
    def __new__(cls, text='unknown', contexts=None, uid=None, toks=None):
        return super().__new__(cls, text, contexts, uid, toks or Toks())


class Context(col.namedtuple('Context', 'text queries uid toks')):
    def __new__(cls, text, queries=None, uid=None, toks=None):
        return super().__new__(cls, text, queries, uid, toks or Toks())


class Query(col.namedtuple('Query', 'text valid uid toks replies possibs')):
    def __new__(cls, text, valid, uid, replies=None, possibs=None, toks=None):
        toks = toks or Toks()
        return super().__new__(cls, text, valid, uid, replies, possibs, toks)


class Reply(col.namedtuple('Reply', 'text span uid toks')):
    def __new__(cls, text, span, uid, toks=None):
        return super().__new__(cls, text, span, uid, toks or Toks())


class Toks(abc.Sequence):

    elems = offsets = ()
    fields = ('_texts', '_spans', '_poss', '_lemmas', ' _ners')
    opts = {}

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, i):
        return self.elems[i]

    def __getstate__(self):
        s = self.__dict__.copy()
        _del_cache(s, self.fields)
        return s

    def __str__(self):
        ts = ' '.join(str(t) for t in self)
        return f'({ts})'

    def reset(self, elems, offsets):
        self.elems = tuple(elems)
        self.offsets = tuple(offsets)
        assert len(self.elems) == len(self.offsets)

    @property
    def texts(self):
        if self._texts is None:
            self._texts = col.Counter(t.text for t in self)
        return self._texts

    @property
    def poss(self):
        if self._poss is None:
            self._poss = col.Counter(t.pos for t in self)
        return self._poss

    @property
    def lemmas(self):
        if self._lemmas is None:
            self._lemmas = col.Counter(t.lemma for t in self)
        return self._lemmas

    @property
    def ners(self):
        if self._ners is None:
            self._ners = col.Counter(t.ner for t in self)
        return self._ners

    @property
    def text(self):
        return ' '.join([t.text for t in self])

    @property
    def groups(self):
        ns = [t.ner for t in self]
        if ns:
            gs = []
            non = self.opts.get('non_ner', 'O')
            i = 0
            while i < len(ns):
                n = ns[i]
                if n == non:
                    i += 1
                else:
                    b = i
                    while (i < len(ns) and ns[i] == n):
                        i += 1
                    gs.append((self.slice(b, i).text(), n))
            return gs

    def slice(self, i=None, j=None):
        ts = copy.copy(self)
        _del_cache(ts.__dict__, self.fields)
        ts._elems = self.elems[i:j]
        return ts

    def ngrams(self, n=1, lower=False, filter_fn=None, as_strings=True):
        ws = [t.text for t in self]
        if lower:
            ws = [w.lower() for w in ws]
        ns = [(s, e + 1) for s in range(len(ws))
              for e in range(s, min(s + n, len(ws)))
              if not filter_fn or not filter_fn(ws[s:e + 1])]
        if as_strings:
            ns = [' '.join(ws[s:e]) for (s, e) in ns]
        return ns


class Topics(abc.Sequence):

    fields = ('_contexts', '_queries', '_replies')

    def __init__(self, elems):
        self.elems = tuple(elems)

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, i):
        return self.elems[i]

    def __getstate__(self):
        s = self.__dict__.copy()
        _del_cache(s, self.fields)
        return s

    def __str__(self):
        ts = ',\n'.join(str(t) for t in self)
        return f'Topics(\n{ts}\n)'

    @property
    def contexts(self):
        if self._contexts is None:
            self._contexts = tuple((t, c) for t in self for c in t.contexts)
        return self._contexts

    @property
    def queries(self):
        if self._queries is None:
            self._queries = tuple(
                (t, c, q) for t, c in self.contexts for q in c.queries)
        return self._queries

    @property
    def replies(self):
        if self._replies is None:
            self._replies = tuple(
                (t, c, q, r) for t, c, q in self.queries for r in q.replies)
        return self._replies

    @property
    def possibs(self):
        if self._possibs is None:
            self._possibs = tuple(
                (t, c, q, p) for t, c, q in self.queries for p in q.possibs)
        return self._possibs


def _span_len(self):
    return self.end - self.beg


Span.__len__ = _span_len


def span__str__(self):
    return f'[{self.beg} {self.end}]'


Span.__str__ = span__str__


def token__str__(self):
    s = f'{self.text} {self.span} {self.lemma} {self.pos} {self.ner}'
    return '{' + s + '}'


Token.__str__ = token__str__


def topic__str__(self):
    cs = ',\n'.join(str(c) for c in self.contexts)
    return f'T("{self.title}"\n({cs})\n)'


Topic.__str__ = topic__str__


def title__str__(self):
    ts = ' '.join(str(t) for t in self.toks)
    return f'Ti("{self.text}"\n({ts})\n)'


Title.__str__ = title__str__


def context__str__(self):
    ts = ' '.join(str(t) for t in self.toks)
    qs = ',\n'.join(str(q) for q in self.queries)
    return f'C("{self.text}"\n({ts})\n({qs})\n)'


Context.__str__ = context__str__


def query__str__(self):
    ts = ' '.join(str(t) for t in self.toks)
    rs = ',\n'.join(str(a) for a in self.replies)
    ps = ',\n'.join(str(v) for v in self.possibs)
    return f'Q("{self.text}" {self.valid}\n({ts})\n({rs})\n({ps})\n)'


Query.__str__ = query__str__


def reply__str__(self):
    ts = ' '.join(str(t) for t in self.toks)
    return f'R("{self.text}" {self.span}\n({ts}))'


Reply.__str__ = reply__str__


def _init_cache(cls, fs, v=None):
    for f in fs:
        setattr(cls, f, v)


_init_cache(Toks, Toks.fields)
_init_cache(Topics, Topics.fields)


def _del_cache(cls, fs):
    for f in fs:
        if hasattr(cls, f):
            delattr(cls, f)
