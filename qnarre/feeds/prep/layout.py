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

import collections as co


def _init_cache(d, fs, v=None):
    for f in fs:
        d[f] = v


def _del_cache(d, fs):
    for f in fs:
        if f in d:
            del d[f]


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


def _span_len(self):
    return self.end - self.begin


Span = co.namedtuple('Span', 'begin end')
Span.__len__ = _span_len

Token = co.namedtuple('Token', 'word span pos lemma ner embeds')


class Tokens(co.abc.Sequence):

    elems = offsets = ()
    fields = ('_words', '_spans', '_poss', '_lemmas', ' _ners')
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
    def words(self):
        if self._words is None:
            self._words = co.Counter(t.word for t in self)
        return self._words

    @property
    def poss(self):
        if self._poss is None:
            self._poss = co.Counter(t.pos for t in self)
        return self._poss

    @property
    def lemmas(self):
        if self._lemmas is None:
            self._lemmas = co.Counter(t.lemma for t in self)
        return self._lemmas

    @property
    def ners(self):
        if self._ners is None:
            self._ners = co.Counter(t.ner for t in self)
        return self._ners

    @property
    def text(self):
        # s = t.beginChar
        # e = ts[i + 1].beginChar if i + 1 < len(ts) else t.endChar
        return ' '.join([t.word for t in self])

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
                    begin = i
                    while (i < len(ns) and ns[i] == n):
                        i += 1
                    gs.append((self.slice(begin, i).text(), n))
            return gs

    def slice(self, i=None, j=None):
        ts = copy.copy(self)
        _del_cache(ts.__dict__, self.fields)
        ts._elems = self.elems[i:j]
        return ts

    def ngrams(self, n=1, lower=False, filter_fn=None, as_strings=True):
        ws = [t.word for t in self]
        if lower:
            ws = [w.lower() for w in ws]
        ns = [(s, e + 1) for s in range(len(ws))
              for e in range(s, min(s + n, len(ws)))
              if not filter_fn or not filter_fn(ws[s:e + 1])]
        if as_strings:
            ns = [' '.join(ws[s:e]) for (s, e) in ns]
        return ns


_init_cache(Tokens.__dict__, Tokens.fields)

Topic = co.namedtuple('Topic', 'title contexts')
# Topic.__new__.__defaults__ = ('', ())

Context = co.namedtuple('Context', 'text tokens questions')
# Context.__new__.__defaults__ = ('', None, ())

Question = co.namedtuple('Question', 'text tokens answers unfit viables qid')
# Question.__new__.__defaults__ = ('', '', None, False, ())

Answer = co.namedtuple('Answer', 'text tokens span uid')


class Topics(co.abc.Sequence):

    fields = ('_contexts', '_questions', '_answers')

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
    def questions(self):
        if self._questions is None:
            self._questions = tuple(
                (t, c, q) for t, c in self.contexts for q in c.questions)
        return self._questions

    @property
    def answers(self):
        if self._answers is None:
            self._answers = tuple(
                (t, c, q, a) for t, c, q in self.questions for a in q.answers)
        return self._answers

    @property
    def viables(self):
        if self._viables is None:
            self._viables = tuple(
                (t, c, q, v) for t, c, q in self.questions for v in q.viables)
        return self._viables


_init_cache(Topics.__dict__, Topics.fields)


def span__str__(self):
    return f'[{self.begin} {self.end}]'


Span.__str__ = span__str__


def token__str__(self):
    s = f'{self.word} {self.span} {self.lemma} {self.pos} {self.ner}'
    return '{' + s + '}'


Token.__str__ = token__str__


def topic__str__(self):
    cs = ',\n'.join(str(c) for c in self.contexts)
    return f'T("{self.title}"\n({cs})\n)'


Topic.__str__ = topic__str__


def context__str__(self):
    ts = ' '.join(str(t) for t in self.tokens)
    qs = ',\n'.join(str(q) for q in self.questions)
    return f'C("{self.text}"\n({ts})\n({qs})\n)'


Context.__str__ = context__str__


def question__str__(self):
    ts = ' '.join(str(t) for t in self.tokens)
    ans = ',\n'.join(str(a) for a in self.answers)
    vs = ',\n'.join(str(v) for v in self.viables)
    return f'Q("{self.text}" {self.unfit}\n({ts})\n({ans})\n({vs})\n)'


Question.__str__ = question__str__


def answer__str__(self):
    ts = ' '.join(str(t) for t in self.tokens)
    return f'A("{self.text}" {self.span}\n({ts}))'


Answer.__str__ = answer__str__
