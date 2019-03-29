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
import enum

import pathlib as pth
import contextlib as cl

from .log import Logger
from .item import Item
from .row import Row
from .node import Node

log = Logger(__name__)


def scanner(path, col):
    rs = []
    with os.scandir(path) as es:
        items = {}
        for e in es:
            p = pth.Path(e.path)
            n = p.stem
            if e.is_dir(follow_symlinks=False):
                rs.append(Node(n))
            elif e.is_file(follow_symlinks=False):
                assert p.suffix
                i = items.get(n)
                if i is None:
                    items[n] = i = Item(path=p)
                    rs.append(Row(n, **{col: i}))
                else:
                    i[p.suffix] = p
            else:
                log.warning('Ignoring dir entry {}', p)
    rs.sort(key=lambda x: x.name)
    for r in rs:
        reject = yield r
        if isinstance(r, Node):
            if reject is not True:
                yield from scanner(path / r.name, col)
                yield None


@cl.contextmanager
def sinker(sink):
    def gen():
        try:
            while True:
                key, entry = yield
                es = sink.setdefault(key, [])
                if entry:
                    es.append(entry)
        except GeneratorExit:
            pass

    g = gen()
    g.send(None)
    yield g


class NoValue(enum.Enum):
    def __repr__(self):
        return "{}.{}".format(type(self).__name__, self.name)


class Sinks(NoValue):

    excluded = enum.auto()
    duplicate = enum.auto()
    synonym = enum.auto()


class Index(dict):
    def __init__(self, tree, **kw):
        super().__init__()
        for e in tree.separator(**kw, uniques=super()):
            pass


class XNames():
    def __init__(self, tree, **kw):
        super().__init__()
        self._tree = tree

    def walker(self, **kw):
        indent = 0
        for e in self._tree.filterer(**kw):
            _, row = e
            if row is None:
                indent -= 2
            else:
                yield ' ' * indent + row.name
                if isinstance(row, Node):
                    indent += 2
        assert indent == 0
