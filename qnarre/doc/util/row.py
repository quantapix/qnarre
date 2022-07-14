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

import asyncio as aio
import collections.abc as abc

from .log import Logger
from .item import Item
from .error import ExtractWarning

log = Logger(__name__)


class Row(abc.MutableMapping):

    _000 = '000'
    _old_name = ''

    def __init__(self, name=_000, cols=None, **kw):
        super().__init__()
        assert not name < self._000
        self.name = name
        self._cols = cols or {}
        self._cols.update(kw)

    def __bool__(self):
        return True

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.name == other.name and self._cols == other._cols)
        return NotImplemented

    def __len__(self):
        return len(self._cols)

    def __iter__(self):
        return iter(self._cols)

    def __getitem__(self, c):
        return self._cols[c]

    def __setitem__(self, c, value):
        self._cols[c] = value

    def __delitem__(self, c):
        del self._cols[c]

    def __repr__(self):
        s = type(self).__name__
        s += "({}".format(repr(self.name))
        s += ", {})".format(repr(self._cols))
        return s

    def stringer(self, indent=0, **kw):
        s = '{} ('.format(self.name)
        for n, e in enumerate(self._cols.items()):
            c, i = e
            gap = ', ' if n else ''
            s += '{}{}: {}'.format(gap, c, i.stringer(**kw) if i else None)
        s += ')'
        yield (" " * indent + s)

    def digest(self, col):
        i = self._cols.get(col)
        return i.digest if i else self.name

    def merge(self, other):
        name = self.name if self.name == other.name else None
        for c, oi in other._cols.items():
            if c in self._cols:
                self._cols[c].merge(oi, name)
            else:
                self._cols[c] = oi

    def rename(self, name):
        self._old_name = self.name
        self.name = name
        log.info('Renaming {} to {}', self._old_name, self.name)

    def extract(self, path, *, base, src_cols=(), ref_cols=(), **kw):
        def _cols():
            for c in ref_cols:
                yield (True, c)
            for c in src_cols:
                yield (False, c)

        i = Item(**kw)
        for ref, c in _cols():
            try:
                if i.expand(self._cols[c], base / c / path, ref):
                    return i
            except KeyError:
                continue
        raise ExtractWarning()

    def touch_item(self, col, path, suff):
        try:
            i = self._cols[col]
        except KeyError:
            self._cols[col] = i = Item()
        i.touch(path, suff)

    def rename_item(self, col, path, to_tmp=False):
        if self._old_name:
            i = self._cols.get(col)
            if i:
                tag = '_qld'
                if to_tmp:
                    path.mkdir(parents=True, exist_ok=True)
                    o = self._old_name
                    i.rename(path, o, o + tag)
                else:
                    i.rename(path, self._old_name + tag, self.name)
                    del self._old_name

    def copy_item(self, col, path, *, base, touch=(), out=None, **_):
        i = self._cols.get(col)
        if i:
            path = path / self.name
            if out:
                i.copy(frm=base / col / path, to=out / col / path)
            elif i.extern_path:

                def _touch_f(suff):
                    for c in touch:
                        self.touch_item(c, base / c / path, suff)

                touch_f = _touch_f if touch else None
                i.copy(to=base / col / path, touch_f=touch_f)

    def clear_col(self, col, **_):
        self._cols.pop(col, None)

    def schedule(self, meth, *args, pool, loop, col=None, **kw):
        tgt = self._cols.get(col) if col else self
        if tgt:
            loop = loop or aio.get_event_loop()
            fut = loop.create_future()

            def _cb(res):
                def safe_cb(res):
                    res = meth(tgt, *args, result=res, **kw)
                    fut.set_result(res)

                loop.call_soon_threadsafe(safe_cb, res)

            def _ecb(exc):
                def safe_ecb(exc):
                    fut.set_exception(exc)

                loop.call_soon_threadsafe(safe_ecb, exc)

            pool.apply_async(meth, (tgt, *args), kw, _cb, _ecb)
            return fut
        return None
