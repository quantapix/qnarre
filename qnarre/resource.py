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

import time

import pprint as pp
import pathlib as pth
import contextlib as cl

from .log import Logger
from .base import config
from .part import Registry

log = Logger(__name__)


@cl.contextmanager
def filelock(path, timeout=None):
    f = FileLock(path, timeout)
    f.acquire()
    yield
    f.release()


class FileLock:

    locked = False

    def __init__(self, path, timeout=None):
        self.lock = path.with_suffix('.lock')
        self.timeout = timeout or 10

    def acquire(self):
        assert not self.locked
        for _ in range(self.timeout):
            try:
                self.lock.touch(exist_ok=False)
            except FileExistsError:
                time.sleep(1)
            else:
                self.locked = True
                return
        raise TimeoutError

    def release(self):
        assert self.locked
        self.lock.unlink()
        del self.locked


def res_path(obj, pref=None):
    p = pth.Path(obj._res_path)
    pref = (obj._realm or pref or config.PROT) + '_'
    return p.with_name(pref + p.name).with_suffix(p.suffix)


class Resource(Registry):

    _realm = config.PROT
    _res_path = None

    @classmethod
    def globals(cls):
        return globals()

    @classmethod
    def create(cls, base, realm=None, **kw):
        b = pth.Path(base)
        p = b / res_path(cls, realm)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            i = eval(p.read_text(), cls.globals())
            i._base = b
            if realm:
                i._realm = realm
            log.info('Restored resource from {}', p)
        else:
            i = cls(**kw, base=b, realm=realm)
        return i

    def __init__(self, elems=None, base=None, realm=None, **kw):
        super().__init__(elems, **kw)
        self._base = base
        if realm:
            self._realm = realm

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._elems == other._elems
        return NotImplemented

    def __repr__(self):
        es = pp.pformat(self._elems, indent=4)
        return '{}({})'.format(type(self).__name__, es)

    @property
    def base(self):
        return self._base

    @property
    def realm(self):
        return self._realm

    @property
    def elems(self):
        es = self._elems
        return (es[k] for k in sorted(es.keys()))

    res_path = res_path

    def rename(self, old, new):
        es = self._elems
        try:
            es[new] = es.pop(old)
        except KeyError:
            pass
            # log.warning('Renaming missing in {} from {} to {}',
            #            type(self), old, new)

    def merge(self, other):
        for k, v in other.items():
            if k in self:
                if self[k] != v:
                    log.info('Values different for {}', k)
            else:
                self[k] = v

    def save(self, pref=None):
        p = self.base / self.res_path(pref)
        if p.exists():
            with filelock(p):
                self.merge(eval(p.read_text(), type(self).globals()))
                p.write_text(repr(self))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(repr(self))


@cl.contextmanager
def resource(r):
    yield r
    r.save()


class Mids(Resource):

    _res_path = config.qnar_dst + 'mids.qnr'

    def __init__(self, elems=None, **kw):
        super().__init__(elems, **kw)
        if not elems:
            for m in config.exclude_mids:
                self._elems[m] = config.EXCLUDED

    def __setitem__(self, mid, name):
        if '|' in mid:
            assert mid == name
        else:
            es = self._elems
            try:
                n = es[mid]
                assert n == name
            except KeyError:
                es[mid] = name

    def rename_msg(self, old, new):
        es = self._elems
        self._elems = {m: n if n != old else new for (m, n) in es.items()}

    def save(self, pref=None):
        super().save(pref)
        ns = {}
        for m, n in self.items():
            assert n
            if n in ns:
                log.warning('MIDs {} and {} with same name {}', m, ns[n], n)
            elif n is not config.EXCLUDED:
                ns[n] = m


class Names(Resource):

    _res_path = 'names.qnr'

    def __init__(self, elems, **kw):
        super().__init__({k: v for k, v in elems.items() if k != v}, **kw)

    def __bool__(self):
        return bool(len(self))
