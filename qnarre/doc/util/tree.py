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

import pathlib as pth
import collections as co
import multiprocessing as mp

from .log import Logger
from .row import Row
from .node import Node
from .utils import sinker, Sinks
from .error import ExtractWarning

log = Logger(__name__)

Entry = co.namedtuple('Entry', 'path row')


class Tree:
    def __init__(self, root, **kw):
        super().__init__()
        if isinstance(root, Node):
            self.name = root.name
            self._root = root
        else:
            self.name = root
            self._root = Node(self.name, **kw)

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.name == other.name and self._root == other._root)
        return NotImplemented

    def __repr__(self):
        s = type(self).__name__
        s += "({})".format(repr(self._root))
        return s

    def stringer(self, indent=0, **kw):
        yield from self._root.stringer(indent=indent + 2, **kw)

    def walker(self, src=None, path=None, col=None, **kw):
        path = path or pth.PurePath()
        if col is not None:
            path /= col
        src = src or self._root.walker(**kw)
        itr = iter(src)
        r = False
        ps = []
        while True:
            r = r or next(itr)
            reject = yield Entry(path, r)
            if reject is True and isinstance(r, Node):
                r = itr.send(reject)
                continue
            if isinstance(r, Node):
                ps.append(path)
                path = path / r.name
            elif r is None:
                path = ps.pop()
            r = False

    def filterer(self, src=None, rejects=None, **kw):
        rejects = rejects or {}
        with sinker(rejects) as sink:
            src = src or self.walker(**kw)

            def _entries(include=None, exclude=None, **kw):
                itr = iter(src)
                e = False
                while True:
                    e = e or next(itr)
                    _, row = e
                    if row is not None:
                        if include is None or not include(e, **kw):
                            if exclude and exclude(e, **kw):
                                sink.send((Sinks.excluded, e))
                                e = itr.send(True)
                                continue
                    yield e
                    e = False

            for e in _entries(**kw):
                yield e

    def appender(self, src, **kw):
        src = self.walker(self._root.appender(src), **kw)
        for e in self.filterer(**kw, src=src):
            yield e

    def extractor(self, col, **kw):
        gen = self.filterer(**kw)
        next(gen)
        for p, r in gen:
            if r is None:
                yield r
            elif isinstance(r, Node):
                yield Node(r.name)
            else:
                try:
                    i = r.extract(p / r.name, **kw)
                    r = Row(r.name, **{col: i})
                    yield r
                except ExtractWarning:
                    log.warning('Extract failed for {}', r.name)

    def separator(self, src=None, duplicates=None, **kw):
        duplicates = duplicates or {}
        with sinker(duplicates) as sink:
            src = src or self.filterer(**kw)

            def _entries(uniques=None, **kw):
                uniques = uniques or {}
                for e in src:
                    path, row = e
                    if row is not None:
                        r = uniques.setdefault(path / row.name, row)
                        if r is row:
                            yield e
                        else:
                            sink.send((Sinks.duplicate, e))

            for e in _entries(**kw):
                yield e

    async def apply(self, meth, *args, src=None, path=None, **kw):
        fs = []
        with mp.Pool() as pool:
            for p, r in src or self.filterer(**kw, path=path):
                if r is not None and not isinstance(r, Node):
                    p /= r.name
                    f = r.schedule(meth, *args, pool=pool, path=p, **kw)
                    if f is not None:
                        fs.append(f)
            for f in fs:
                await f
        return all(f.result() for f in fs)

    def normalize(self, col_of=None, **kw):
        col, cols = col_of or (None, ())
        self._root.consolidate(col)
        self._root.normalize(bool(cols))
        for c in cols:
            self.rename_items(c, **kw)

    def rower(self, **kw):
        for e in self.filterer(**kw):
            _, r = e
            if r is not None and not isinstance(r, Node):
                yield e

    def rename_items(self, col, *, base, **kw):
        for p, r in self.rower(**kw, col=col):
            r.rename_item(col, pth.Path(base / p), True)
        for p, r in self.rower(**kw, col=col):
            r.rename_item(col, pth.Path(base / p))

    def copy_items(self, col, **kw):
        for p, r in self.rower(**kw):
            r.copy_item(col, p, **kw)

    def clear_col(self, col, **kw):
        for _, r in self.rower(**kw):
            r.clear_col(col, **kw)

    def stats(self, **kw):
        dirs, files = 0
        for e in self.filterer(**kw):
            _, r = e
            if isinstance(r, Node):
                dirs += 1
            elif isinstance(r, Row):
                files += 1
        return (dirs, files)
