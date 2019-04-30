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

import regex as R
import pathlib as P


class Splitter:
    def __init__(self, lower_case=True):
        self.lower_case = lower_case

    def __call__(self, txt, offset=0):
        off = 0
        for w in unicodedata.normalize('NFD', txt).split():
            if not w or not w.isprintable():
                continue
            off = txt.find(w, off)
            if self.lower_case:
                w = w.lower()
            if not w.isascii():
                w = ''.join(f' {c} ' if self.is_chinese(c) else c for c in w)
                w = ''.join('' if self.is_accent(c) else c for c in w)
            if w.isalnum():
                yield w, offset + off
            else:
                lcs, los, new = [], [], True
                for c in list(w):
                    if self.is_punct(c):
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
                    yield ''.join(cs), offset + o

    def join(self, splits, offsets):
        pass

    def test(self, txt):
        pass

    @staticmethod
    def is_chinese(char):
        n = ord(char)
        return ((n >= 0x2B820 and n <= 0x2CEAF)
                or (n >= 0x2A700 and n <= 0x2B73F)
                or (n >= 0x3400 and n <= 0x4DBF)
                or (n >= 0x20000 and n <= 0x2A6DF)
                or (n >= 0xF900 and n <= 0xFAFF)
                or (n >= 0x2B740 and n <= 0x2B81F)
                or (n >= 0x4E00 and n <= 0x9FFF)
                or (n >= 0x2F800 and n <= 0x2FA1F))

    @staticmethod
    def is_accent(char):
        return unicodedata.category(char) == 'Mn'

    @staticmethod
    def is_punct(char):
        n = ord(char)
        if ((n >= 33 and n <= 47) or (n >= 58 and n <= 64)
                or (n >= 91 and n <= 96) or (n >= 123 and n <= 126)):
            return True
        return unicodedata.category(char).startswith('P')


PAD, UNK, CLS, SEP, MASK, SOS, EOS = ('[PAD]', '[UNK]', '[CLS]', '[SEP]',
                                      '[MASK]', '[SOS]', '[EOS]')


class BertEncoder(Splitter):
    max_chars = 200
    _by_ids = None

    @classmethod
    def load(cls, params, **kw):
        PS = params
        p = P.Path(PS.model_dir) / PS.model_name
        with open(p / 'vocab.txt', mode='rt') as f:
            v = {t.strip(): i for i, t in enumerate(f)}
        PS.update(PAD=v[PAD], UNK=v[UNK], CLS=v[CLS], SEP=v[SEP], MASK=v[MASK])
        PS.update(vocab=v, vocab_size=len(v))
        lc = PS.lower_case or PS.model_name.startswith('uncased')
        return cls(v, lower_case=lc, **kw)

    def __init__(self, vocab, max_chars=None, **kw):
        super().__init__(**kw)
        self.vocab = vocab
        if max_chars:
            self.max_chars = max_chars

    def __call__(self, txt, offset=0):
        unk = self.vocab[UNK]
        for w, offset in super()(txt, offset):
            cs = list(w)
            if len(cs) > self.max_chars:
                yield unk, offset
            else:
                b = 0
                while b < len(cs):
                    e, unk = len(cs), True
                    while b < e:
                        s = '##' if b > 0 else ''
                        s += ''.join(cs[b:e])
                        if s in self.vocab:
                            yield self.vocab[s], offset + b
                            unk = False
                            break
                        e -= 1
                    if unk:
                        yield unk, offset + b
                        return
                    b = e

    @property
    def by_ids(self):
        if self._by_ids is None:
            ts = sorted(self.vocab.items(), key=lambda _, i: i)
            self._by_ids = [t for t, _ in ts]
        return self._by_ids

    def decode(self, ids, offsets):
        pass

    def test(self, txt):
        pass


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


class BPEncoder(Splitter):

    from_byte, from_code = _bytes_to_code()
    pattern = R.compile(_pat)
    _by_ids = None

    @classmethod
    def load(cls, params):
        PS = params
        p = P.Path(PS.model_dir)
        with lzma.open(p / 'tokens.json.xz', mode='rt') as f:
            v = json.load(f)
        PS.vocab_size = len(v)
        with lzma.open(p / 'pairs.bpe.xz', mode='rt', encoding='utf-8') as f:
            pairs = f.read()
        pairs = tuple(tuple(p.split()) for p in pairs.split('\n')[1:-1])
        return cls(v, pairs, lower_case=PS.lower_case)

    def __init__(self, vocab, pairs, **kw):
        super().__init__(**kw)
        self.vocab = vocab
        self.pairs = dict(zip(pairs, range(len(pairs))))
        self.cache = {}

    def __call__(self, txt, offset=0):
        unk = self.vocab[UNK]
        for w, offset in super()(txt, offset):
            o = 0
            for t in R.findall(self.pattern, w):
                o = w.find(t, o)
                sw = ''.join(self.from_byte[b] for b in t.encode())
                for st in self.segment(sw):
                    assert o + len(st) < len(w)
                    yield self.vocab.get(st, unk), offset + o
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

    @property
    def by_ids(self):
        if self._by_ids is None:
            ts = sorted(self.vocab.items(), key=lambda _, i: i)
            self._by_ids = [t for t, _ in ts]
        return self._by_ids

    def decode(self, ids, offsets):
        ts = ''.join(self.by_ids[i] for i in ids)
        bs = bytearray([self.from_code[c] for c in ts])
        return bs.decode(errors='replace')

    def test(self, txt):
        pass


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
