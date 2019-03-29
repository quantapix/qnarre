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

# import pytest

import shutil as sh
import pathlib as pth

from qnarre.roster import Roster


def create_file(base, path):
    (base / path).write_text('{}:{}'.format(path.parent.name, path.name))


def create_files():
    b = pth.Path('/tmp/qn-roster')
    if b.exists():
        sh.rmtree(str(b))
    u = 'u'
    m = 'm'
    for p in ('a', 'a/a', 'a/b', 'a/b/a', 'b'):
        (b / u / p).mkdir(parents=True, exist_ok=True)
        (b / m / p).mkdir(parents=True, exist_ok=True)
    for p in ('c', 'c/a', 'c/b/a'):
        (b / m / p).mkdir(parents=True, exist_ok=True)
    d = 'd'
    (b / d).mkdir(parents=True, exist_ok=True)

    for p in ('a', 'a/a', 'a/b', 'a/b/a', 'b'):
        p = pth.Path(p)
        for f in ('010.a', '020.b', '030'):
            create_file(b / u, p / f)
        create_file(b / m, p / '020.b')
        create_file(b / m, p / '040.c')
    for p in ('c', 'c/b/a'):
        p = pth.Path(p)
        for f in ('010.a', '030'):
            create_file(b / m, p / f)


def test_roster_1():
    create_files()
    b = pth.Path('/tmp/qn-roster')
    rp = b / '.roster.qnr'
    if rp.exists():
        rp.unlink()
    r = Roster.create(base=b)
    u = 'u'
    r.scan((b / u,))
    assert r.check_ok()
    s = repr(r)
    assert s == "Roster((Entry('u/a/010.a', 'c8f0cea60efad43cf583659a5e11abf994c4af43', 7), Entry('u/a/020.b', '5f2b7b8e8d543a38ab933f5daeca47b27f50b529', 7), Entry('u/a/030', 'e1be6a7f2ff38a976bfbe87de63be79fabc95160', 5), Entry('u/b/010.a', '8494a5bc3a51eeaa3c7679010c47e956f9e88d04', 7), Entry('u/b/020.b', '31b4013e7477d6e411a735a90d95757e4a4b54ac', 7), Entry('u/b/030', '86a539320c2df647dce458c15056582ae6f011d7', 5)))"
    r.save(rp)
    r2 = Roster.create(b)
    assert r2.check_ok()
    assert r == r2
    (b / u / 'a/020.b').write_text('aaa')
    assert not r.check_ok()
    create_file(b / u, pth.Path('a/020.b'))
    if rp.exists():
        rp.unlink()


def test_roster_2():
    b = pth.Path('/tmp/qn-roster')
    rp = b / '.roster.qnr'
    if rp.exists():
        rp.unlink()
    r = Roster.create(base=b)
    u = 'u'
    r.scan((b / u,))
    assert r.check_ok()
    # print(r)
    r.save()
    r = Roster.create(base=b)
    assert r.check_ok()
    m = 'm'
    r.scan((b / m,))
    # print('***')
    # print(r)
    assert r.check_ok()
    s = repr(r)
    assert s == "Roster((Entry('m/a/040.c', '5ad14582f29b9b009ce5a802b51e557ba7d7b1a6', 7), Entry('m/b/040.c', '5b5fb87905ee80cdcbd8c472917e589f939bba6a', 7), Entry('m/c/010.a', '3c19c836960f376934fd469b335241a74d358b68', 7), Entry('m/c/030', '0baa5dd5b9ce891ac40b19d4e891e75dd4f737c3', 5), Entry('u/a/010.a', 'c8f0cea60efad43cf583659a5e11abf994c4af43', 7), Entry('u/a/020.b', '5f2b7b8e8d543a38ab933f5daeca47b27f50b529', 7), Entry('u/a/030', 'e1be6a7f2ff38a976bfbe87de63be79fabc95160', 5), Entry('u/b/010.a', '8494a5bc3a51eeaa3c7679010c47e956f9e88d04', 7), Entry('u/b/020.b', '31b4013e7477d6e411a735a90d95757e4a4b54ac', 7), Entry('u/b/030', '86a539320c2df647dce458c15056582ae6f011d7', 5)))"
    d = 'd'
    r.expel(b / d)


def test_roster_3():
    b = pth.Path('/tmp/qn-roster/qbase')
    b.mkdir(parents=True, exist_ok=True)
    rp = b / 'roster.qnr'
    if rp.exists():
        rp.unlink()
    r = Roster.create(base=b)
    u = 'u'
    r.absorb(u, b.parent)
    r.absorb(u, b.parent)
    m = 'm'
    r.absorb(m, b.parent)
    r.absorb(m, b.parent)
    d = 'd'
    r.absorb(d, b.parent)
    r.absorb(d, b.parent)
    r.absorb(d, b.parent)
