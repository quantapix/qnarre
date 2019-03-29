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

import collections as co
import collections.abc as abc

from .row import Row
from .log import Logger
from .base import num_to_name

log = Logger(__name__)


class Node(Row, abc.MutableSequence):
    def __init__(self, name, rows=None, **kw):
        super().__init__(name, **kw)
        self._rows = [Row(**kw)] if rows is None else [*rows]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (super().__eq__(other) and self._rows == other._rows)
        return NotImplemented

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, str):
            return super()[i]
        else:
            return self._rows[i]

    def __setitem__(self, i, value):
        if isinstance(i, str):
            super()[i] = value
        else:
            self._rows[i] = value

    def __delitem__(self, i):
        if isinstance(i, str):
            del super()[i]
        else:
            del self._rows[i]

    def insert(self, i, value):
        return self._rows.insert(i, value)

    def __iadd__(self, other):
        if isinstance(other, Row):
            self._rows.append(other)
        elif other is not None:
            self._rows.extend(other)
        return self

    def __repr__(self):
        s = type(self).__name__
        s += "({}, (".format(repr(self.name))
        for r in self._rows:
            s += repr(r) + ", "
        s += "), cols={})".format(repr(self._cols))
        return s

    def stringer(self, indent=0, **kw):
        yield (" " * indent + self.name + ":")
        for r in self._rows:
            yield from r.stringer(indent + 2, **kw)

    def walker(self, depth_first=False, **_):
        def _breadth_first():
            nodes = []
            for r in self._rows:
                if isinstance(r, Node):
                    nodes.append(r)
                else:
                    yield r
            for r in nodes:
                yield r

        reject = yield self
        if reject is not True:
            for r in self._rows if depth_first else _breadth_first():
                if isinstance(r, Node):
                    yield from r.walker(depth_first)
                else:
                    yield r
            yield None

    def appender(self, src, itr=None):
        rs = []
        itr = itr or iter(src)
        r = False
        try:
            while r is not None:
                r = r or next(itr)
                if r is None:
                    break
                reject = yield r
                if reject is True:
                    r = itr.send(reject)
                    continue
                rs.append(r)
                if isinstance(r, Node):
                    yield from r.appender(src, itr)
                    yield None
                r = False
        except StopIteration:
            pass
        finally:
            self._rows.extend(rs)

    def merge(self, other):
        assert isinstance(other, Node)
        super().merge(other)
        self._rows.extend(other._rows)

    rename = None

    def consolidate(self, col):
        rs = co.OrderedDict()
        for r in self._rows:
            d = r.digest(col)
            try:
                rs[d].merge(r)
            except KeyError:
                rs[d] = r
        self._rows = list(rs.values())
        for r in self._rows:
            if isinstance(r, Node):
                r.consolidate(col)

    def normalize(self, rename):
        self._rows.sort(key=lambda x: x.name)
        dirty = False
        i = 0
        for r in self._rows:
            if isinstance(r, Node):
                r.normalize(rename)
            elif rename:
                n = num_to_name(i)
                if r.name != n:
                    r.rename(n)
                    dirty = True
                i += 1
        if dirty:
            self._rows.sort(key=lambda x: x.name)

    schedule = None
