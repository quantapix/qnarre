# Copyright 2018 Quantapix Authors. All Rights Reserved.
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
import pytest

import pathlib as pth

from pathlib import PosixPath
from qnarre.item import Item
from qnarre.row import Row
from qnarre.node import Node
from qnarre.tree import Tree
from qnarre.utils import Index
from qnarre.table import Table


ptrn_1 = {
    '1.u    ': {'a2':  9, 'a1':  9, 'c2': '', 'c1': '', 'b2': '', 'b1': ''},
    '2.u    ': {'a2':  9, 'a1':  9, 'c2': '', 'c1': 11, 'b2': 11, 'b1': 11},
    '3.u    ': {'a2':  9, 'a1': 11, 'c2': '', 'c1':  0, 'b2': 11, 'b1': 11},

    'a/1.u  ': {'a2': 11, 'a1': '', 'c2': '', 'c1':  0, 'b2': 11, 'b1': 11},
    'a/2.u  ': {'a2':  9, 'a1':  9, 'c2': 12, 'c1': 11, 'b2': 12, 'b1': 11},
    'a/3.u  ': {'a2':  9, 'a1':  9, 'c2': 11, 'c1': '', 'b2': 11, 'b1': ''},

    'a/a/1.u': {'a2': 11, 'a1': '', 'c2': 12, 'c1':  0, 'b2': 12, 'b1': 11},
    'a/a/2.u': {'a2':  9, 'a1': 11, 'c2':  0, 'c1': '', 'b2': 11, 'b1': ''},
    'a/a/3.u': {'a2': 12, 'a1': '', 'c2':  0, 'c1': 11, 'b2': 12, 'b1': 11},

    'a/b/1.u': {'a2':  9, 'a1': 11, 'c2':  0, 'c1':  0, 'b2': 11, 'b1': 11},

    'a/b/2.u': {'a2':  9, 'a1':  9, 'c2': '', 'c1': 11, 'b2': 11, 'b1': 11},
    'a/b/2.v': {'a2':  9, 'a1':  9, 'c2': '', 'c1': 11, 'b2': 11, 'b1': 11},

    'a/b/3.u': {'a2':  9, 'a1': 11, 'c2': '', 'c1':  0, 'b2': 11, 'b1': 11},
    'a/b/3.v': {'a2':  9, 'a1':  9, 'c2': '', 'c1': 11, 'b2': 11, 'b1': 11},

    'b/a/1.u': {'a2': 11, 'a1': '', 'c2': 12, 'c1':  0, 'b2': 12, 'b1': 11},
    'b/a/1.v': {'a2':  9, 'a1': 11, 'c2': '', 'c1':  0, 'b2': '', 'b1': 11},

    'b/a/2.u': {'a2':  9, 'a1':  9, 'c2': 12, 'c1': 11, 'b2': 12, 'b1': 11},
    'b/a/2.v': {'a2':  9, 'a1':  9, 'c2': 12, 'c1': 11, 'b2': 12, 'b1': 11},

    'b/a/3.u': {'a2':  9, 'a1':  9, 'c2': 11, 'c1': '', 'b2': 11, 'b1': ''},
    'b/a/3.v': {'a2': 11, 'a1': '', 'c2':  0, 'c1':  9, 'b2': 11, 'b1': ''},

    'b/b/1.u': {'a2': 12, 'a1': '', 'c2':  0, 'c1': 11, 'b2': 12, 'b1': 11},
    'b/b/1.v': {'a2':  9, 'a1':  9, 'c2': '', 'c1': 11, 'b2': 11, 'b1': 11},
}


def fix_pattern(ptrn):
    r = {}
    for p, cs in ptrn.items():
        r[p.strip()] = cs
    return r


ptrn_1 = fix_pattern(ptrn_1)


def dump_pattern(ptrn, base, col, tree='', encode=True):
    for p, cs in ptrn.items():
        if tree:
            p = base / col / tree / p
        else:
            p = base / col / p
        p.parent.mkdir(parents=True, exist_ok=True)
        t = p.name
        if encode:
            sf = p.suffix
            p = p.with_name('{:0>2s}0'.format(p.stem))
            p = p.with_suffix(sf)
        v = cs[col]
        if v == '':
            if p.exists():
                p.unlink()
        elif v == 0:
            p.touch(exist_ok=True)
        else:
            if v == 9:
                t = '{:*<{size}}'.format(t, size=v)
            else:
                t = '{:<{size}}'.format(t, size=v)
            p.write_text(t)


def load_pattern(ptrn, base, col, tree='', decode=True):

    def gen(path):
        with os.scandir(path) as es:
            for e in es:
                p = pth.Path(e.path)
                if e.is_dir(follow_symlinks=False):
                    yield from gen(p)
                elif e.is_file(follow_symlinks=False):
                    t = p.read_text()
                    s = p.stat().st_size
                    if tree:
                        p = p.relative_to(base / col / tree)
                    else:
                        p = p.relative_to(base / col)
                    if decode:
                        sf = p.suffix
                        p = p.with_name(p.stem[1:-1])
                        p = p.with_suffix(sf)
                    if s == 9:
                        assert t == '{:*<{size}}'.format(p.name, size=s)
                    elif s > 9:
                        assert t == '{:<{size}}'.format(p.name, size=s)
                    else:
                        assert s == 0
                    yield (str(p), s)

    for p, s in gen(base / col / tree if tree else base / col):
        try:
            ptrn[p][col] = s
        except KeyError:
            ptrn[p] = {col: s}


def load_missing(ptrn, col):
    for p in ptrn.keys():
        cs = ptrn[p]
        if col not in cs:
            cs[col] = ''


def test_pattern():
    b = pth.Path('/tmp/qn-test4')
    p1 = ptrn_1
    dump_pattern(p1, b, 'a2', encode=False)
    for c in ('a1', 'c2', 'c1', 'b2', 'b1'):
        dump_pattern(p1, b, c, 't1')
    p2 = {}
    load_pattern(p2, b, 'a2', decode=False)
    for c in ('a1', 'c2', 'c1', 'b2', 'b1'):
        load_pattern(p2, b, c, 't1')
    for c in ('a1', 'c2', 'c1', 'b2', 'b1'):
        load_missing(p2, c)
    assert p1 == p2


@pytest.mark.asyncio
async def test_layout_1(event_loop=None):
    b = pth.Path('/tmp/qn-test5')
    b.mkdir(parents=True, exist_ok=True)
    ts = ('t1', 't2')
    cs = ('a2', 'a1', 'c2', 'c1', 'b2', 'b1')
    sc = ('a1', 'a2')
    t = Table(base=b, trees=ts, cols=cs)
    p = pth.Path('/tmp/qn-test4')
    await t.import_rows(p / 'a2', trees='t1', loop=event_loop)
    await t.load_cols(cs[1:-2], p, trees='t1', loop=event_loop)
    await t.extract('b1', suffs=('.u',), src_cols=sc, ref_cols=('c1',),
                    trees='t1', loop=event_loop)
    await t.extract('b2', suffs=('.u',), src_cols=sc, ref_cols=('c2', 'c1'),
                    trees='t1', loop=event_loop)
    print('\nTable start...')
    for s in t.stringer():
        print(s)
    print('... end Table')
    p = pth.Path('/tmp/qn-test6')
    p.mkdir(parents=True, exist_ok=True)
    t.dump_cols(p)
    ptrn_2 = {}
    load_pattern(ptrn_2, p, 'a2', 't1')
    for c in ('a1', 'c2', 'c1', 'b2', 'b1'):
        load_pattern(ptrn_2, p, c, 't1')
    for c in ('a1', 'c2', 'c1', 'b2', 'b1'):
        load_missing(ptrn_2, c)
    # for p, cs in ptrn_1.items():
        # print(p, sorted(cs.items()))
        # print(p, sorted(ptrn_2[p].items()))
    assert ptrn_1 == ptrn_2
