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

import json

import regex as re
import pathlib as pth
import collections as col

from qnarre.feeds.prep import utils


def tokenizer_for(ps):
    t = ps.tokenizer
    if t == 'char':
        return CharEncoder(ps)
    elif t == 'bert':
        return BertEncoder(ps)
    elif t == 'gpt_2':
        return Gpt2Encoder(ps)
    assert t is None or t == 'word'
    return WordEncoder(ps)


class Splitter:
    def __init__(self, lower_case=False):
        self.lower_case = lower_case

    def __call__(self, txt, offset=0):
        txt = txt.lower() if self.lower_case else txt
        off = 0
        for w in txt.split():
            if not w or not w.isprintable():
                continue
            off = txt.find(w, off)
            if not w.isascii():
                w = ''.join(f' {c} ' if utils.is_chinese(c) else c for c in w)
                w = ''.join('' if utils.is_accent(c) else c for c in w)
            if w.isalnum():
                yield w, offset + off
                off += len(w)
            else:
                lcs, los, new = [], [], True
                for c in list(w):
                    if utils.is_punct(c):
                        lcs.append([c])
                        los.append(off)
                        new = True
                    else:
                        if new:
                            lcs.append([])
                            los.append(off)
                            new = False
                        lcs[-1].append(c)
                    off += 1
                for cs, o in zip(lcs, los):
                    w = ''.join(cs)
                    yield w, offset + o


class SplitCounter(Splitter):
    def __init__(self, lower_case=False):
        super().__init__(lower_case)
        self.count = col.Counter()

    def __call__(self, txt, offset=0):
        w, o = super()(txt, offset)
        self.count.update(w)
        return w, o


def join_splits(splits, offsets):
    i, ts = 0, []
    for s, o in zip(splits, offsets):
        if i < o:
            ts.append(' ' * (o - i))
            i = o
        else:
            assert i == o
        ts.append(s)
        i += len(s)
    return ''.join(ts)


class WordEncoder:
    def __init__(self, ps, vocab=None):
        self.ps = ps
        if vocab is None:
            v = pth.Path(ps.dir_data) / ps.dset / ps.vocab_path
            if v.exists():
                vocab = v
        self.vocab = utils.Vocab(ps, vocab)
        lc = ps.lower_case
        if lc is None:
            lc = ps.model.startswith('uncased')
        self.splitter = Splitter(lc)

    def __call__(self, txt, offset=0):
        maxc = self.ps.tok_max_chars or 200
        for w, o in self.splitter(txt, offset):
            if w in self.vocab:
                t = self.vocab[w]
            elif len(w) > maxc or self.vocab.fixed:
                t = self.ps.UNK
            else:
                t = self.vocab.append(w)
            yield t, o, w

    def decode(self, ids, offsets):
        return join_splits((self.vocab[i] for i in ids), offsets)


class CharEncoder(WordEncoder):
    def __call__(self, txt, offset=0):
        for w, o in self.splitter(txt, offset):
            for i, c in enumerate(list(w)):
                t = self.vocab.append(c)
                yield t, o + i, c


class BertEncoder(WordEncoder):
    def __init__(self, ps):
        v = pth.Path(ps.dir_data) / ps.dset / ps.vocab_path
        if not v.exists():
            with open(ps.bert_vocab, mode='rt') as f:
                v = f.readlines()
        super().__init__(ps, v)

    def __call__(self, txt, offset=0):
        maxc = self.ps.tok_max_chars or 200
        for w, o in self.splitter(txt, offset):
            if len(w) > maxc:
                yield self.ps.UNK, o, w
            else:
                cs = list(w)
                b = 0
                while b < len(cs):
                    e, unk = len(cs), True
                    while b < e:
                        s = '##' if b > 0 else ''
                        s += ''.join(cs[b:e])
                        if s in self.vocab:
                            yield self.vocab[s], o + b, s
                            unk = False
                            break
                        e -= 1
                    if unk:
                        yield self.ps.UNK, o + b, ''.join(cs[b:e])
                        return
                    b = e

    def decode(self, ids, offsets):
        def splits():
            for i in ids:
                s = self.vocab[i]
                yield s[2:] if s.startswith('##') else s

        return join_splits(splits(), offsets)


class Gpt2Encoder(WordEncoder):
    pat = r"'s|'t|'re|'ve|'m|'ll|'d|"
    pat += r' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'
    pat = re.compile(pat)
    from_byte, from_code = utils.bytes_to_code()

    def __init__(self, ps):
        v = pth.Path(ps.dir_data) / ps.dset / ps.vocab_path
        if not v.exists():
            with open(ps.gpt_2_vocab, mode='rt') as f:
                ts = sorted(json.load(f).items(), key=lambda i: i[1])
            v = []
            for i, (w, j) in enumerate(ts):
                assert i == j
                v.append(w)
        super().__init__(ps, v)
        with open(ps.gpt_2_pairs, mode='rt', encoding='utf-8') as f:
            data = f.read()
        ts = tuple(tuple(d.split()) for d in data.splitlines()[1:])
        self.pairs = dict(zip(ts, range(len(ts))))
        self.cache = {}

    def __call__(self, txt, offset=0):
        b = ''
        for w, off in self.splitter(txt, offset):
            o, w = 0, b + w
            for t in re.findall(self.pat, w):
                o = w.find(t, o)
                sw = ''.join(self.from_byte[b] for b in t.encode())
                for st in self.segment(sw):
                    assert o + len(st) <= len(w)
                    yield self.vocab.get(st, self.ps.UNK), off + o, st
                    o += len(st)
            b = ' '

    def segment(self, word):
        if word in self.cache:
            return self.cache[word]
        segs = tuple(word)
        while len(segs) > 1:

            def min_pair():
                ps = set()
                f = segs[0]
                for s in segs[1:]:
                    ps.add((f, s))
                    f = s
                p = min(ps, key=lambda p: self.pairs.get(p, float('inf')))
                if p in self.pairs:
                    return p

            p = min_pair()
            if p is None:
                break
            lf, rt = p
            ss, i = [], 0
            while i < len(segs):
                try:
                    j = segs.index(lf, i)
                    ss.extend(segs[i:j])
                    i = j
                except ValueError:
                    ss.extend(segs[i:])
                    break
                if segs[i] == lf and i < len(segs) - 1 and segs[i + 1] == rt:
                    ss.append(lf + rt)
                    i += 2
                else:
                    ss.append(segs[i])
                    i += 1
            segs = tuple(ss)
        self.cache[word] = segs
        return segs

    def decode(self, ids, _):
        ts = ''.join(self.vocab[i] for i in ids)
        bs = bytearray([self.from_code[c] for c in ts])
        return bs.decode(errors='replace')
