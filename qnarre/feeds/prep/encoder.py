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

from collections import Counter, OrderedDict

from tensorflow.gfile import Open as open
from tensorflow.gfile import Exists as exists


class Vocab(object):
    def __init__(self,
                 special=[],
                 min_freq=0,
                 max_size=None,
                 lower_case=True,
                 delimiter=None,
                 vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print('counting file {} ...'.format(path))
        assert exists(path)

        sents = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('  line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
      sents : a list of sentences, each a list of tokenized symbols
    """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('  line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def encode_file(self,
                    path,
                    ordered=False,
                    verbose=False,
                    add_eos=True,
                    add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert exists(path)
        encoded = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('  line {}'.format(idx))
                symbols = self.tokenize(line,
                                        add_eos=add_eos,
                                        add_double_eos=add_double_eos)
                encoded.append(self.convert_to_nparray(symbols))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('  line {}'.format(idx))
            encoded.append(self.convert_to_nparray(symbols))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_nparray(self, symbols):
        nparray = np.array(self.get_indices(symbols), dtype=np.int64)
        return nparray

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join(
                [self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)


class OpenAIVocab(Vocab):
    def __init__(self, max_size, vocab_file=None):
        from pytorch_pretrained_bert import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.EOT = self.tokenizer.encoder['<|endoftext|>']
        self.max_size = max_size
        self.vocab_file = vocab_file

    def __len__(self):
        return len(self.tokenizer)

    def count_file(self, path, verbose=False, add_eos=False):
        pass

    def build_vocab(self):
        pass

    def encode_file(self,
                    path,
                    ordered=False,
                    verbose=False,
                    add_eos=True,
                    add_double_eos=False) -> torch.LongTensor:
        cached = path + '.tokenized'
        if os.path.exists(cached):
            print('found cache')
            return torch.load(cached)
        print(f'encoding file {path} ...')
        assert os.path.exists(path), f"{path} doesn't exist"

        with open(path, encoding='utf-8') as f:
            # Suppress warnings about length.
            with open(os.devnull,
                      "w") as devnull, contextlib.redirect_stderr(devnull):
                out = torch.LongTensor(
                    self.tokenizer.encode(f.read()) + [self.EOT])
                with portalocker.Lock(cached, timeout=60) as _:
                    torch.save(out, cached)
                return out


class GoogleBPEVocab(Vocab):
    """Don't use this until this issue is fixed.

    https://github.com/google/sentencepiece/issues/318
    """

    def __init__(self, max_size, vocab_file=None):
        import sentencepiece as spm
        self.spm = spm
        self.max_size = max_size
        self.vocab_file = vocab_file
        self.sp = spm.SentencePieceProcessor()

    def count_file(self, path, verbose=False, add_eos=False):
        self.spm.SentencePieceTrainer.Train(
            f'--input={self.vocab_file} --model_prefix=m --vocab_size={self.max_size} --model_type=bpe'
        )

    def build_vocab(self):
        if self.vocab_file:
            self.sp.Load(self.vocab_file)
        else:
            pass

    def encode_file(self,
                    path,
                    ordered=False,
                    verbose=False,
                    add_eos=True,
                    add_double_eos=False) -> torch.LongTensor:
        with open(path, encoding='utf-8') as f:
            return torch.LongTensor(self.sp.EncodeAsIds(f.read()))
