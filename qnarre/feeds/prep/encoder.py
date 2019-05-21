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
import lzma
import unicodedata

import regex as re
import pathlib as pth

from collections import Counter


class Splitter:
    def __init__(self, ps):
        self.ps = ps
        self.count = Counter()

    def __call__(self, txt, offset=0):
        off = 0
        for w in unicodedata.normalize('NFD', txt).split():
            if not w or not w.isprintable():
                continue
            off = txt.find(w, off)
            if self.ps.tok_lower_case:
                w = w.lower()
            if not w.isascii():
                w = ''.join(f' {c} ' if is_chinese(c) else c for c in w)
                w = ''.join('' if is_accent(c) else c for c in w)
            if w.isalnum():
                self.count.update(w)
                yield w, offset + off
            else:
                lcs, los, new = [], [], True
                for c in list(w):
                    if is_punct(c):
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
                    self.count.update(w)
                    yield w, offset + o

    def join(self, splits, offsets):
        i, ts = 0, []
        for s, o in zip(splits, offsets):
            if i < o:
                ts.append(' ' * (o - i))
                i = o
            else:
                assert i == o
            ts.append(s)
        return ''.join(ts)


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


CLS = '[CLS]'
EOS = '[EOS]'
MSK = '[MSK]'
PAD = '[PAD]'
SEP = '[SEP]'
SOS = '[SOS]'
UNK = '[UNK]'


class Encoder:
    _by_ids = None

    @classmethod
    def load(cls, ps):
        p = pth.Path(ps.model_dir) / ps.model_name
        with open(p / 'vocab.txt', mode='rt') as f:
            v = {t.strip(): i for i, t in enumerate(f)}
        ps.update(
            num_toks=len(v),
            tok_vocab=v,
            CLS=v[CLS],
            EOS=v[EOS],
            MSK=v[MSK],
            PAD=v[PAD],
            SEP=v[SEP],
            SOS=v[SOS],
            UNK=v[UNK],
        )
        if ps.tok_lower_case is None:
            ps.update(tok_lower_case=ps.model_name.startswith('uncased'))
        return cls(ps)

    def __init__(self, ps):
        self.ps = ps
        self.splitter = Splitter(ps)

    def __call__(self, txt, offset=0):
        unk = self.ps.tok_vocab[UNK]
        for w, offset in self.splitter(txt, offset):
            cs = list(w)
            if self.ps.tok_max_chars and len(cs) > self.ps.tok_max_chars:
                yield unk, offset, cs
            else:
                b = 0
                while b < len(cs):
                    e, unk = len(cs), True
                    while b < e:
                        s = '##' if b > 0 else ''
                        s += ''.join(cs[b:e])
                        if s in self.ps.tok_vocab:
                            yield self.ps.tok_vocab[s], offset + b, s
                            unk = False
                            break
                        e -= 1
                    if unk:
                        yield unk, offset + b, cs[b:e]
                        return
                    b = e

    @property
    def by_ids(self):
        if self._by_ids is None:
            ts = sorted(self.ps.tok_vocab.items(), key=lambda _, i: i)
            self._by_ids = [t for t, _ in ts]
        return self._by_ids

    def decode(self, ids, offsets):
        return self.splitter.join((self.by_ids[i] for i in ids), offsets)


class BertEncoder(Encoder):
    def __call__(self, txt, offset=0):
        unk = self.ps.tok_vocab[UNK]
        for _, offset, w in self.splitter(txt, offset):
            cs = list(w)
            if self.ps.tok_max_chars and len(cs) > self.ps.tok_max_chars:
                yield unk, offset, cs
            else:
                b = 0
                while b < len(cs):
                    e, unk = len(cs), True
                    while b < e:
                        s = '##' if b > 0 else ''
                        s += ''.join(cs[b:e])
                        if s in self.ps.tok_vocab:
                            yield self.ps.tok_vocab[s], offset + b, s
                            unk = False
                            break
                        e -= 1
                    if unk:
                        yield unk, offset + b, cs[b:e]
                        return
                    b = e


def _bytes_to_code():
    bc = {b: chr(b) for b in range(ord("!"), ord("~") + 1)}
    bc.update({b: chr(b) for b in range(ord("¡"), ord("¬") + 1)})
    bc.update({b: chr(b) for b in range(ord("®"), ord("ÿ") + 1)})
    i = 0
    for b in range(2**8):
        if b not in bc:
            bc[b] = chr(2**8 + i)
            i += 1
    return bc, {c: b for b, c in bc.items()}


_pat = r"'s|'t|'re|'ve|'m|'ll|'d|"
_pat += r' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'


class BPEncoder(Encoder):
    from_byte, from_code = _bytes_to_code()
    pattern = re.compile(_pat)

    @classmethod
    def load(cls, ps):
        p = pth.Path(ps.model_dir)
        with lzma.open(p / 'tokens.json.xz', mode='rt') as f:
            v = json.load(f)
        ps.update(
            num_toks=len(v),
            tok_vocab=v,
            CLS=v[CLS],
            EOS=v[EOS],
            MSK=v[MSK],
            PAD=v[PAD],
            SEP=v[SEP],
            SOS=v[SOS],
            UNK=v[UNK],
        )
        with lzma.open(p / 'pairs.bpe.xz', mode='rt', encoding='utf-8') as f:
            pairs = f.read()
        pairs = tuple(tuple(p.split()) for p in pairs.split('\n')[1:-1])
        return cls(ps, pairs)

    def __init__(self, ps, pairs):
        super().__init__(ps)
        self.pairs = dict(zip(pairs, range(len(pairs))))
        self.cache = {}

    def __call__(self, txt, offset=0):
        unk = self.ps.tok_vocab[UNK]
        for w, offset in self.splitter(txt, offset):
            o = 0
            for t in re.findall(self.pattern, w):
                o = w.find(t, o)
                sw = ''.join(self.from_byte[b] for b in t.encode())
                for st in self.segment(sw):
                    assert o + len(st) < len(w)
                    yield self.ps.tok_vocab.get(st, unk), offset + o
                    o += len(st)

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

            p = self.min_pair()
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

    def decode(self, ids, offsets):
        ts = ''.join(self.by_ids[i] for i in ids)
        bs = bytearray([self.from_code[c] for c in ts])
        return bs.decode(errors='replace')


"""
def normalize(txt):
    txt = txt.replace('—', '-')
    txt = txt.replace('–', '-')
    txt = txt.replace('―', '-')
    txt = txt.replace('…', '...')
    txt = txt.replace('´', "'")
    txt = R.sub(
        r'(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)',
        r' \1 ', txt)
    txt = R.sub(r'\s*\n\s*', ' \n ', txt)
    txt = R.sub(r'[^\S\n]+', ' ', txt)
    return txt.strip()
"""
