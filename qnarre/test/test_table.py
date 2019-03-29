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

import sys
import pytest

import pathlib as pth

from pathlib import PosixPath
from qnarre.item import Item
from qnarre.row import Row
from qnarre.node import Node
from qnarre.tree import Tree
from qnarre.utils import Index
from qnarre.table import Table


def test_table_create_1():
    t = Table(trees=('tree_a', 'tree_b'), cols=('col_a', 'col_b'), kw_1=True)
    s = repr(t)
    assert s == "Table(None, {'tree_a': Tree(Node('tree_a', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_b': Tree(Node('tree_b', (Row('000', {'kw_1': True}), ), cols={'kw_1': True}))}, ('col_a', 'col_b'))"
    t2 = eval(s)
    assert s == repr(t2)
    assert t == t2


def test_table_create_2():
    p = pth.Path('/tmp/qn-test')
    for c in ('col_0', 'col_1', 'col_2'):
        for t in ('tree_0', 'tree_1', 'tree_2'):
            (p / c / t).mkdir(parents=True, exist_ok=True)
    t = Table(p, trees=('tree_a', 'tree_b'), cols='col_a', kw_1=True)
    s = repr(t)
    assert s == "Table('/tmp/qn-test', {'tree_a': Tree(Node('tree_a', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_b': Tree(Node('tree_b', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_1': Tree(Node('tree_1', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_2': Tree(Node('tree_2', (Row('000', {'kw_1': True}), ), cols={'kw_1': True}))}, ('col_a', 'col_0', 'col_1', 'col_2'))"
    t2 = eval(s)
    assert s == repr(t2)
    assert t == t2


def test_table_create_3():
    p = pth.Path('/tmp/qn-test')
    for c in ('col_0', 'col_1', 'col_2'):
        for t in ('tree_0', 'tree_1', 'tree_2'):
            (p / c / t).mkdir(parents=True, exist_ok=True)
    t = Table(p, trees='tree_a', kw_1=True)
    s = repr(t)
    assert s == "Table('/tmp/qn-test', {'tree_a': Tree(Node('tree_a', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_1': Tree(Node('tree_1', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_2': Tree(Node('tree_2', (Row('000', {'kw_1': True}), ), cols={'kw_1': True}))}, ('col_0', 'col_1', 'col_2'))"
    t2 = eval(s)
    assert s == repr(t2)
    assert t == t2


def test_table_create_4():
    p = pth.Path('/tmp/qn-test')
    for c in ('col_0', 'col_1', 'col_2'):
        for t in ('tree_0', 'tree_1', 'tree_2'):
            (p / c / t).mkdir(parents=True, exist_ok=True)
    t = Table(p, kw_1=True)
    s = repr(t)
    assert s == "Table('/tmp/qn-test', {'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_1': Tree(Node('tree_1', (Row('000', {'kw_1': True}), ), cols={'kw_1': True})), 'tree_2': Tree(Node('tree_2', (Row('000', {'kw_1': True}), ), cols={'kw_1': True}))}, ('col_0', 'col_1', 'col_2'))"
    t2 = eval(s)
    assert s == repr(t2)
    assert t == t2


@pytest.mark.asyncio
async def test_table_import_1(event_loop=None):
    p = pth.Path('/tmp/qn-src')
    t = 'tree_0'

    (p / t / 'na' / 'naa' / 'naaa').mkdir(parents=True, exist_ok=True)
    (p / t / 'na' / 'naa' / 'naab').mkdir(parents=True, exist_ok=True)
    (p / t / 'na' / 'nab').mkdir(parents=True, exist_ok=True)
    (p / t / 'nb').mkdir(parents=True, exist_ok=True)

    (p / t / 'na' / 'naa' / 'naaa' / 'raaa_1.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'naa' / 'naaa' / 'raaa_2.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'naa' / 'naaa' / 'raaa_3.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'naa' / 'naab' / 'raab_1.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'naa' / 'raa_1.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'naa' / 'raa_2.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'nab' / 'rab_1.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'nab' / 'rab_2.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'ra_1.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'ra_2.pdf').touch(exist_ok=True)
    (p / t / 'na' / 'ra_3.pdf').touch(exist_ok=True)
    (p / t / 'nb' / 'rb_1.pdf').touch(exist_ok=True)
    (p / t / 'nb' / 'rb_2.pdf').touch(exist_ok=True)
    (p / t / 'r_1.pdf').touch(exist_ok=True)
    (p / t / 'r_2.pdf').touch(exist_ok=True)
    (p / t / 'r_3.pdf').touch(exist_ok=True)

    tb = Table(trees=t, cols=('col_0', 'col_1'), kw_1=True)
    b = pth.Path('/tmp/qn-test2')
    b.mkdir(parents=True, exist_ok=True)
    try:
        await tb.import_rows(p / t, base=b, loop=event_loop)
        s = repr(tb)
        assert s == "Table(None, {'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Node('na', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Node('naa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Node('naaa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), Node('naab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), ), cols={}), Node('nab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), ), cols={}), Node('nb', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), ), cols={'kw_1': True}))}, ('col_0', 'col_1'))"
        tb2 = eval(s)
        assert s == repr(tb2)
        assert tb == tb2
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


@pytest.mark.asyncio
async def test_table_import_2(event_loop=None):
    p = pth.Path('/tmp/qn-src2')
    t = 'tree_0'

    (p / t / 'na' / 'naa' / 'naaa').mkdir(parents=True, exist_ok=True)
    (p / t / 'na' / 'naa' / 'naab').mkdir(parents=True, exist_ok=True)
    (p / t / 'na' / 'nab').mkdir(parents=True, exist_ok=True)
    (p / t / 'nb').mkdir(parents=True, exist_ok=True)

    def create_file(path):
        path.write_text(str(path))

    create_file(p / t / 'na' / 'naa' / 'naaa' / 'raaa_1.pdf')
    create_file(p / t / 'na' / 'naa' / 'naaa' / 'raaa_2.pdf')
    create_file(p / t / 'na' / 'naa' / 'naaa' / 'raaa_2.rst')
    create_file(p / t / 'na' / 'naa' / 'naaa' / 'raaa_3.pdf')
    create_file(p / t / 'na' / 'naa' / 'naab' / 'raab_1.pdf')
    create_file(p / t / 'na' / 'naa' / 'raa_1.pdf')
    create_file(p / t / 'na' / 'naa' / 'raa_2.pdf')
    create_file(p / t / 'na' / 'naa' / 'raa_2.rst')
    create_file(p / t / 'na' / 'nab' / 'rab_1.pdf')
    create_file(p / t / 'na' / 'nab' / 'rab_2.pdf')
    create_file(p / t / 'na' / 'nab' / 'rab_2.rst')
    create_file(p / t / 'na' / 'ra_1.pdf')
    create_file(p / t / 'na' / 'ra_2.pdf')
    create_file(p / t / 'na' / 'ra_2.rst')
    create_file(p / t / 'na' / 'ra_3.pdf')
    create_file(p / t / 'nb' / 'rb_1.pdf')
    create_file(p / t / 'nb' / 'rb_2.pdf')
    create_file(p / t / 'nb' / 'rb_2.rst')
    create_file(p / t / 'r_1.pdf')
    create_file(p / t / 'r_2.pdf')
    create_file(p / t / 'r_2.rst')
    create_file(p / t / 'r_3.pdf')

    tb = Table(trees=t, cols=('col_0', 'col_1'), kw_1=True)
    b = pth.Path('/tmp/qn-test2')
    b.mkdir(parents=True, exist_ok=True)
    await tb.import_rows(p / t, base=b, loop=event_loop)
    s = repr(tb)
    assert s == "Table(None, {'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), Row('010', {'col_0': Item(('.pdf',), '64326c7c23bf67ec1bbba676c122aaca42301de6')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '9b51cff4d7b70fe17377ffaec8d89bcad3122fa6')}), Row('030', {'col_0': Item(('.pdf',), 'bb20037c427096464cb480a3376f2650c0bc0a61')}), Node('na', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'b0774d713aef8f4be85c5b3a01eaef0660910995')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '12fc7a304fcf698d638824d556d9f439ceba3d00')}), Row('030', {'col_0': Item(('.pdf',), '54db64757aa079398ca9837e669867f9422f6e67')}), Node('naa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9e8310d002d4ab11ad491c6561ccc7c346f6f94a')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '02f56370a4bc94b895915e47b83612919afa3c80')}), Node('naaa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '58c37912f8ce7a006e797241715790a545de8bcd')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '04332491fb92f8d777db9419c9e61cc143c1fe83')}), Row('030', {'col_0': Item(('.pdf',), 'b6dceaa055fd2fe5dd2d2fb3b4f8adf349e923e2')}), ), cols={}), Node('naab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9ffd12d8db034cd970739c986f4bb240cb630378')}), ), cols={}), ), cols={}), Node('nab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '2b9a022a924ffe817148afc3b878783d8e9e671a')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '3fc7d5076daf9f1e9de327de8809dde7eeeaa148')}), ), cols={}), ), cols={}), Node('nb', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '94946c5dc2aa9fbf184ca0528f95012e0657c452')}), Row('020', {'col_0': Item(('.pdf', '.rst'), 'abd6f9afc464e213d5fd336be266748ae3b729b2')}), ), cols={}), ), cols={'kw_1': True}))}, ('col_0', 'col_1'))"
    tb2 = eval(s)
    assert s == repr(tb2)
    assert tb == tb2


@pytest.mark.asyncio
async def test_table_import_3(event_loop=None):
    p = pth.Path('/tmp/qn-src3')
    t = 'tree_0'

    (p / t / 'c1' / 'c1a').mkdir(parents=True, exist_ok=True)
    (p / t / 'c1' / 'c1b').mkdir(parents=True, exist_ok=True)
    (p / t / 'c2' / 'c2c').mkdir(parents=True, exist_ok=True)
    (p / t / 'c2' / 'c2d').mkdir(parents=True, exist_ok=True)

    def create_file(path):
        path.write_text(str(path))

    create_file(p / t / 'c1' / 'r_1.pdf')
    create_file(p / t / 'c1' / 'r_2.pdf')
    create_file(p / t / 'c1' / 'c1a' / 'ra_1.pdf')
    create_file(p / t / 'c1' / 'c1a' / 'ra_2.pdf')
    create_file(p / t / 'c1' / 'c1a' / 'ra_3.pdf')
    create_file(p / t / 'c1' / 'c1b' / 'rb_1.pdf')
    create_file(p / t / 'c2' / 'r_3.pdf')
    create_file(p / t / 'c2' / 'r_4.pdf')
    create_file(p / t / 'c2' / 'c2c' / 'rc_1.pdf')
    create_file(p / t / 'c2' / 'c2d' / 'rd_1.pdf')

    tb = Table(trees=t, cols=('col_0', 'col_1', 'col_2'))
    b = pth.Path('/tmp/qn-test3')
    b.mkdir(parents=True, exist_ok=True)
    await tb.import_rows(p / t / 'c1', base=b, touch=('col_1',), loop=event_loop)
    await tb.import_rows(p / t / 'c2', base=b, touch=('col_2',), loop=event_loop)
    s = repr(tb)
    assert s == "Table(None, {'tree_0': Tree(Node('tree_0', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'b1cf43e1c2787342fb3aec634dd7f8df4d3bb1b9'), 'col_1': Item(('.pdf',))}), Row('020', {'col_0': Item(('.pdf',), 'e78ce5aa9080dd8c8e2124fc80bf50c68d9cf255'), 'col_1': Item(('.pdf',))}), Row('030', {'col_0': Item(('.pdf',), 'c9965541c8fec0c0125f92a1682e0e55c65ac77f'), 'col_2': Item(('.pdf',))}), Row('040', {'col_0': Item(('.pdf',), '4a2b2b4693c3963af00798b0d3ff27716ab01e93'), 'col_2': Item(('.pdf',))}), Node('c1a', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'ec34d278a2219d9c20a40cbc1a0a0967f67b2015'), 'col_1': Item(('.pdf',))}), Row('020', {'col_0': Item(('.pdf',), '317470086900d07a0437be3dcdc3af5117c7914a'), 'col_1': Item(('.pdf',))}), Row('030', {'col_0': Item(('.pdf',), '7c34f2bb9cefb1805a919d26aa5a2b7903b46107'), 'col_1': Item(('.pdf',))}), ), cols={}), Node('c1b', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '7f9da70bfafcf87d1212100ad29cf9ae6ea3b535'), 'col_1': Item(('.pdf',))}), ), cols={}), Node('c2c', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '543b8e74a7245e97ad84bfad80d3ea4761f208ec'), 'col_2': Item(('.pdf',))}), ), cols={}), Node('c2d', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '98d9418b6625404de81e544b1418781c1cb8374f'), 'col_2': Item(('.pdf',))}), ), cols={}), ), cols={}))}, ('col_0', 'col_1', 'col_2'))"
    tb2 = eval(s)
    assert s == repr(tb2)
    assert tb == tb2


@pytest.mark.asyncio
async def test_table_load_cols_1(event_loop=None):
    try:
        b = pth.Path('/tmp/qn-test2')
        tb = Table(b, kw_1=True)
        await tb.load_cols(loop=event_loop)
        await tb.load_cols(loop=event_loop)
        await tb.load_cols(loop=event_loop)
        s = repr(tb)
        assert s == "Table('/tmp/qn-test2', {'tree_0': Tree(Node('tree_0', (Row('000', {'kw_1': True}), Row('010', {'col_0': Item(('.pdf',), '64326c7c23bf67ec1bbba676c122aaca42301de6')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '9b51cff4d7b70fe17377ffaec8d89bcad3122fa6')}), Row('030', {'col_0': Item(('.pdf',), 'bb20037c427096464cb480a3376f2650c0bc0a61')}), Node('na', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'b0774d713aef8f4be85c5b3a01eaef0660910995')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '12fc7a304fcf698d638824d556d9f439ceba3d00')}), Row('030', {'col_0': Item(('.pdf',), '54db64757aa079398ca9837e669867f9422f6e67')}), Node('naa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9e8310d002d4ab11ad491c6561ccc7c346f6f94a')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '02f56370a4bc94b895915e47b83612919afa3c80')}), Node('naaa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '58c37912f8ce7a006e797241715790a545de8bcd')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '04332491fb92f8d777db9419c9e61cc143c1fe83')}), Row('030', {'col_0': Item(('.pdf',), 'b6dceaa055fd2fe5dd2d2fb3b4f8adf349e923e2')}), ), cols={}), Node('naab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9ffd12d8db034cd970739c986f4bb240cb630378')}), ), cols={}), ), cols={}), Node('nab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '2b9a022a924ffe817148afc3b878783d8e9e671a')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '3fc7d5076daf9f1e9de327de8809dde7eeeaa148')}), ), cols={}), ), cols={}), Node('nb', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '94946c5dc2aa9fbf184ca0528f95012e0657c452')}), Row('020', {'col_0': Item(('.pdf', '.rst'), 'abd6f9afc464e213d5fd336be266748ae3b729b2')}), ), cols={}), ), cols={'kw_1': True}))}, ('col_0',))"
        tb2 = eval(s)
        assert s == repr(tb2)
        assert tb == tb2
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


@pytest.mark.asyncio
async def test_table_check_items(event_loop=None):
    b = pth.Path('/tmp/qn-test2')
    tb = Table(b)
    await tb.load_cols(loop=event_loop)
    assert await tb.check_items(loop=event_loop)
    s = repr(tb)
    tb2 = eval(s)
    assert await tb2.check_items(loop=event_loop)
    p = b / 'col_0' / 'tree_0' / 'na' / '010.pdf'
    c = p.read_text()
    p.write_text('abra-cadabra')
    assert not await tb2.check_items(loop=event_loop)
    p.write_text(c)
    assert await tb2.check_items(loop=event_loop)
    p = p.with_suffix('.rst')
    p.write_text(str(p))
    try:
        assert await tb2.check_items(loop=event_loop)
        await tb2.load_cols(loop=event_loop)
        assert await tb2.check_items(loop=event_loop)
        s2 = repr(tb2)
        tb3 = eval(s2)
        assert s2 == repr(tb3)
        assert tb2 == tb3
        assert await tb3.check_items(loop=event_loop)
    except:
        raise
    finally:
        p.unlink()
    assert not await tb2.check_items(loop=event_loop)
    tb4 = Table(b)
    await tb4.load_cols(loop=event_loop)
    assert await tb4.check_items(loop=event_loop)
    assert s == repr(tb4)


@pytest.mark.asyncio
async def test_table_load_1(event_loop=None):
    b = pth.Path('/tmp/qn-test2')
    tb = Table(b, cols=('col_0', 'col_1', 'col_2'))
    await tb.load_cols('col_0', loop=event_loop)
    assert await tb.check_items(loop=event_loop)
    idx = Index(tb['tree_0'])
    p1 = b / 'col_1' / 'tree_0' / 'na' / '010.rst'
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text('aaa')
    p2 = b / 'col_2' / 'tree_0' / 'na' / 'naa' / '010.rst'
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text('bbb')
    try:
        await tb.load_cols(loop=event_loop)
        assert await tb.check_items(loop=event_loop)
        idx2 = Index(tb['tree_0'])
        assert idx == idx2
        p3 = p1.with_name('011').with_suffix('.rst')
        p3.write_text('aaa')
        p4 = p2.with_name('011').with_suffix('.rst')
        p4.write_text('bbb')
        try:
            await tb.load_cols(loop=event_loop)
            assert await tb.check_items(loop=event_loop)
            idx3 = Index(tb['tree_0'])
            assert not idx == idx3
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            p3.unlink()
            p4.unlink()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        p1.unlink()
        p2.unlink()
        import os
        os.removedirs(b / 'col_1' / 'tree_0' / 'na')
        os.removedirs(b / 'col_2' / 'tree_0' / 'na' / 'naa')
    s = repr(tb)
    assert s == "Table('/tmp/qn-test2', {'tree_0': Tree(Node('tree_0', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '64326c7c23bf67ec1bbba676c122aaca42301de6')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '9b51cff4d7b70fe17377ffaec8d89bcad3122fa6')}), Row('030', {'col_0': Item(('.pdf',), 'bb20037c427096464cb480a3376f2650c0bc0a61')}), Node('na', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'b0774d713aef8f4be85c5b3a01eaef0660910995'), 'col_1': Item(('.rst',), '38cfd717aaa25f278785ef096102394ffabe62b1')}), Row('011', {'col_1': Item(('.rst',), '38cfd717aaa25f278785ef096102394ffabe62b1')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '12fc7a304fcf698d638824d556d9f439ceba3d00')}), Row('030', {'col_0': Item(('.pdf',), '54db64757aa079398ca9837e669867f9422f6e67')}), Node('naa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9e8310d002d4ab11ad491c6561ccc7c346f6f94a'), 'col_2': Item(('.rst',), 'c94759d4ede95ce375d4cd2ced1b933e75e8d0c3')}), Row('011', {'col_2': Item(('.rst',), 'c94759d4ede95ce375d4cd2ced1b933e75e8d0c3')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '02f56370a4bc94b895915e47b83612919afa3c80')}), Node('naaa', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '58c37912f8ce7a006e797241715790a545de8bcd')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '04332491fb92f8d777db9419c9e61cc143c1fe83')}), Row('030', {'col_0': Item(('.pdf',), 'b6dceaa055fd2fe5dd2d2fb3b4f8adf349e923e2')}), ), cols={}), Node('naab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '9ffd12d8db034cd970739c986f4bb240cb630378')}), ), cols={}), ), cols={}), Node('nab', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '2b9a022a924ffe817148afc3b878783d8e9e671a')}), Row('020', {'col_0': Item(('.pdf', '.rst'), '3fc7d5076daf9f1e9de327de8809dde7eeeaa148')}), ), cols={}), ), cols={}), Node('nb', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '94946c5dc2aa9fbf184ca0528f95012e0657c452')}), Row('020', {'col_0': Item(('.pdf', '.rst'), 'abd6f9afc464e213d5fd336be266748ae3b729b2')}), ), cols={}), ), cols={}))}, ('col_0', 'col_1', 'col_2'))"
    tb2 = eval(s)
    assert tb == tb2
    assert s == repr(tb2)
    assert not await tb.check_items(loop=event_loop)


@pytest.mark.asyncio
async def test_table_load_2(event_loop=None):
    try:
        b = pth.Path('/tmp/qn-test3')
        tb = Table(b)
        await tb.load_cols(loop=event_loop)
        await tb.load_cols(loop=event_loop)
        await tb.load_cols(loop=event_loop)
        print('\nTable start...')
        for l in tb.stringer():
            print(l)
        print('... end Table')
        s = repr(tb)
        assert s == "Table('/tmp/qn-test3', {'tree_0': Tree(Node('tree_0', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'b1cf43e1c2787342fb3aec634dd7f8df4d3bb1b9'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Row('020', {'col_0': Item(('.pdf',), 'e78ce5aa9080dd8c8e2124fc80bf50c68d9cf255'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Row('030', {'col_0': Item(('.pdf',), 'c9965541c8fec0c0125f92a1682e0e55c65ac77f'), 'col_2': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Row('040', {'col_0': Item(('.pdf',), '4a2b2b4693c3963af00798b0d3ff27716ab01e93'), 'col_2': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Node('c1a', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), 'ec34d278a2219d9c20a40cbc1a0a0967f67b2015'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Row('020', {'col_0': Item(('.pdf',), '317470086900d07a0437be3dcdc3af5117c7914a'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), Row('030', {'col_0': Item(('.pdf',), '7c34f2bb9cefb1805a919d26aa5a2b7903b46107'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), Node('c1b', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '7f9da70bfafcf87d1212100ad29cf9ae6ea3b535'), 'col_1': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), Node('c2c', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '543b8e74a7245e97ad84bfad80d3ea4761f208ec'), 'col_2': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), Node('c2d', (Row('000', {}), Row('010', {'col_0': Item(('.pdf',), '98d9418b6625404de81e544b1418781c1cb8374f'), 'col_2': Item(('.pdf',), '3345524abf6bbe1809449224b5972c41790b6cf2')}), ), cols={}), ), cols={}))}, ('col_0', 'col_1', 'col_2'))"
        tb2 = eval(s)
        assert s == repr(tb2)
        assert tb == tb2
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
