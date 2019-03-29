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

import shutil as sh
import pathlib as pth
import collections.abc as abc

from hashlib import blake2b

from .log import Logger

log = Logger(__name__)


class Item(abc.MutableMapping):

    _digest = ''

    def __init__(self, suffs=(), digest='', path='', **_):
        super().__init__()
        self._suffs = dict.fromkeys(suffs)
        if path:
            path = pth.Path(path)
            self._suffs[path.suffix] = path
        if digest:
            self._digest = digest

    def __bool__(self):
        return True

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.suffs == other.suffs
                    and self._digest == other._digest)
        return NotImplemented

    def __len__(self):
        return len(self._suffs)

    def __iter__(self):
        return iter(self._suffs)

    def __getitem__(self, s):
        return self._suffs[s]

    def __setitem__(self, s, value):
        self._suffs[s] = value

    def __delitem__(self, s):
        del self._suffs[s]

    def __repr__(self):
        s = type(self).__name__
        s += '({}'.format(repr(tuple(self.suffs)))
        if self._digest:
            s += ', {}'.format(repr(self._digest))
        s += ')'
        return s

    def stringer(self, digest=None, **_):
        s = '('
        for n, suff in enumerate(self.suffs):
            s += '{}{}'.format(', ' if n else '', suff)
        s += ')'
        if self._digest:
            s += ' {}'.format(self._digest if digest else '#..')
        return s

    @property
    def name(self):
        for p in self.paths():
            return p.stem
        return ''

    @property
    def suffs(self):
        return sorted([s for s in self._suffs.keys() if s])

    def paths(self, path=None):
        ps = None if path is None else path.suffix
        path = self._suffs.get('') if path is None else path
        for s, p in sorted(self._suffs.items(), key=lambda x: x[0]):
            if s:
                if p:
                    assert s == p.suffix
                    yield p
                elif path:
                    if ps:
                        assert s == ps
                        yield path
                    else:
                        yield path.with_suffix(s)

    @property
    def extern_path(self):
        for p in self.paths():
            return True
        return False

    @property
    def digest(self):
        d = self._digest
        if not d:
            raise ValueError
        return d

    def calc_digest(self, path=None, check=False, result=None, **_):
        if result is None:
            digest = blake2b(digest_size=20)
            for p in self.paths(path):
                if p.exists():
                    size = 0
                    with open(p, 'rb') as f:
                        for d in iter(lambda: f.read(65536), b''):
                            size += len(d)
                            digest.update(d)
                    assert size == p.stat().st_size
            digest = digest.hexdigest()
            if check:
                return digest == self._digest
            self._digest = digest
            return digest
        elif isinstance(result, str):
            self._digest = result
            return True
        else:
            return result

    def touch(self, path, suff):
        path = path.with_suffix(suff)
        if suff not in self._suffs or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self._suffs[suff] = None

    def rename(self, path, frm, to):
        frm = path.with_name(frm)
        to = path.with_name(to)
        for s in self.suffs:
            f = frm.with_suffix(s)
            if f.exists():
                f.rename(to.with_suffix(s))

    def merge(self, other, name):
        d = self._digest
        od = other._digest
        assert od
        if d != od:
            if d:
                log.warning('Overwriting digest {}', name)
            self._digest = od
        for s, op in other._suffs.items():
            assert op
            p = self._suffs.get(s)
            if p and d != od:
                assert p == op
            else:
                self._suffs[s] = op

    def copy(self, frm=None, to=None, touch_f=None):
        def _copy(frm, to):
            if frm.exists():
                if not to.exists() or not frm.samefile(to):
                    to.parent.mkdir(parents=True, exist_ok=True)
                    sh.copyfile(str(frm), str(to))
                return True
            return False

        if frm is None:
            for f in self.paths():
                s = f.suffix
                if _copy(f, to.with_suffix(s)) and touch_f:
                    touch_f(s)
            for s in self._suffs.keys():
                if s:
                    self._suffs[s] = None
                else:
                    del self._suffs[s]
        else:
            for s in self.suffs:
                _copy(frm.with_suffix(s), to.with_suffix(s))

    def probe(self, path, suff):
        if suff in self._suffs:
            path = path.with_suffix(suff)
            if path.exists():
                return True if path.stat().st_size == 0 else path
        return False

    def expand(self, other, path, ref):
        def _try_add(s, p=True):
            o = other.probe(path, s)
            if o and (ref or (p and isinstance(o, pth.Path))):
                self._suffs[s] = o

        for s, p in self._suffs.items():
            if s and not isinstance(p, pth.Path):
                if not p or not ref:
                    _try_add(s, p)
        if ref:
            for s in other._suffs.keys():
                if s and s not in self._suffs:
                    _try_add(s)
        return all(isinstance(p, pth.Path) for p in self._suffs.values())
