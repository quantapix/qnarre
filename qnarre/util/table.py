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

import os

import pathlib as pth
import collections.abc as abc

from .log import Logger
from .item import Item
from .tree import Tree
from .utils import scanner

log = Logger(__name__)

calc_dig = Item.calc_digest


class Table(abc.MutableMapping):
    def __init__(self, base=None, trees=None, cols=None, **kw):
        super().__init__()
        self.base = base = pth.Path(base) if base else None
        if not isinstance(cols, tuple):
            cols = () if cols is None else (cols, )

        def _cols():
            cs = frozenset(cols)
            ns = []
            with os.scandir(base) as scan:
                for e in scan:
                    if e.is_dir(follow_symlinks=False):
                        n = pth.Path(e.path).stem
                        if n not in cs:
                            ns.append(n)
            ns.sort()
            return (*cols, *ns)

        self._cols = cols = _cols() if base is not None else cols
        assert self._cols

        if isinstance(trees, dict):
            self._trees = trees
        else:
            if not isinstance(trees, tuple):
                trees = () if trees is None else (trees, )
            if base is not None:

                def _trees():
                    ts = frozenset(trees)
                    ns = []
                    for c in cols:
                        p = base / c
                        if p.exists():
                            with os.scandir(p) as scan:
                                for e in scan:
                                    if e.is_dir(follow_symlinks=False):
                                        n = pth.Path(e.path).stem
                                        if n not in ts:
                                            ns.append(n)
                    ns.sort()
                    return (*trees, *ns)

                trees = _trees()
            self._trees = {n: Tree(n, **kw) for n in trees}
        assert self._trees

    def __bool__(self):
        return True

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self._trees == other._trees and self._cols == other._cols)
        return NotImplemented

    def __len__(self):
        return len(self._trees)

    def __iter__(self):
        return iter(self._trees)

    def __getitem__(self, n):
        return self._trees[n]

    def __setitem__(self, n, tree):
        self._trees[n] = tree

    def __delitem__(self, n):
        del self._trees[n]

    def __repr__(self):
        s = type(self).__name__
        b = str(self.base) if self.base else None
        s += "({}".format(repr(b))
        s += ", {}".format(repr(self._trees))
        s += ", {})".format(repr(self._cols))
        return s

    def stringer(self, indent=0, **kw):
        for t in self._trees.values():
            yield from t.stringer(indent=indent, **kw)

    def walker(self, trees=None, cols=None, **_):
        if trees is None:
            trees = self._trees.values()
        else:
            trees = trees if isinstance(trees, tuple) else (trees, )
            trees = [self._trees[t] for t in trees if t in self._trees]
        if not trees:
            log.warning('Empty trees list')
        if cols is None:
            cols = self._cols
        else:
            cols = cols if isinstance(cols, tuple) else (cols, )
            cols = [c for c in cols if c in self._cols]
        if not cols:
            log.warning('Empty cols list')
        for t in trees:
            for c in cols:
                yield (t, c)

    def adjust_kw(self, kw):
        kw['base'] = base = kw.get('base') or self.base
        assert base.exists() and base.is_dir()
        try:
            kw['touch'] = [c for c in kw['touch'] if c in self._cols]
        except KeyError:
            pass
        return base

    async def import_rows(self, src, cols=None, **kw):
        self.adjust_kw(kw)
        for t, c in self.walker(**kw, cols=cols or self._cols[0]):
            s = src / c if isinstance(cols, tuple) else src
            s = t.appender(scanner(s, c), **kw, col=c)
            await t.apply(calc_dig, **kw, src=s, col=c)
            t.normalize((c, self._cols), **kw)
            t.copy_items(c, **kw)

    async def load_cols(self, cols=None, src=None, **kw):
        base = self.adjust_kw(kw)
        src = src or base
        for t, c in self.walker(**kw, cols=cols):
            s = src / c / t.name
            s = t.appender(scanner(s, c), **kw, col=c)
            await t.apply(calc_dig, **kw, src=s, col=c)
            t.normalize(**kw)
            t.copy_items(c, **kw)

    def dump_cols(self, dst, cols=None, **kw):
        self.adjust_kw(kw)
        for t, c in self.walker(**kw, cols=cols):
            t.copy_items(c, **kw, out=dst)

    def clear_cols(self, cols=None, **kw):
        self.adjust_kw(kw)
        for t, c in self.walker(**kw, cols=cols):
            t.clear_col(c, **kw)

    async def check_items(self, **kw):
        kw['path'] = self.adjust_kw(kw)
        for t, c in self.walker(**kw):
            if not await t.apply(calc_dig, check=True, **kw, col=c):
                return False
        return True

    async def extract(self, col=None, **kw):
        self.adjust_kw(kw)
        col = col if col in self._cols else self._cols[-1]
        for t, c in self.walker(**kw, cols=col):
            s = t.extractor(**kw, col=c)
            s = t.appender(s, **kw, col=c)
            await t.apply(calc_dig, **kw, src=s, col=c)
            t.normalize(**kw)
            t.copy_items(c, **kw)
