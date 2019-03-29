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

import re
import pprint as pp

from .log import Logger
from .base import config
from .resource import Resource

from .date import Date  # needed for repr reading

log = Logger(__name__)


class AdrFilter:
    def __init__(self, incl=(), doms=(), locs=(), fulls=()):
        super().__init__()

        def init(ds, ss):
            return {k: True if k in ss else False for k in set((*ds, *ss))}

        self.incl = init(config.include_adrs, incl)
        self.doms = init(config.exclude_doms, doms)
        self.locs = init(config.exclude_locs, locs)
        self.fulls = init(config.exclude_fulls, fulls)

    def __repr__(self):
        s = '{}('.format(type(self).__name__)

        def keys(es):
            return pp.pformat(tuple(sorted(k for k, v in es if v)), indent=4)

        s += '{}, '.format(keys(self.incl.items()))
        s += '{}, '.format(keys(self.doms.items()))
        s += '{}, '.format(keys(self.locs.items()))
        s += '{})'.format(keys(self.fulls.items()))
        return s

    def probe(self, adr):
        if adr in self.incl:
            self.incl[adr] = True
            return True
        if adr in self.fulls:
            self.fulls[adr] = True
            return False
        ps = adr.split('@')
        if len(ps) == 2:
            l, ad = ps
            ds = ad.split('.')
            d2 = ds[-2] + '.' + ds[-1]
            for d in (d2, ad):
                if d in self.incl:
                    self.incl[d] = True
                    return True
            if l in self.incl:
                self.incl[l] = True
                return True
            for d in ('.' + ds[-1], d2, ad):
                if d in self.doms:
                    self.doms[d] = True
                    return False
            if l in self.locs:
                self.locs[l] = True
                return False
        else:
            log.info('Invalid address {}', adr)


class RAdrFilter:
    def __init__(self, spec, **_):
        super().__init__()
        self.spec = spec
        self._cspec = re.compile(spec, re.ASCII)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.spec)

    def probe(self, adr):
        if self._cspec.match(adr):
            return False


class Filters(Resource):

    _res_path = config.qnar_dst + 'filts/filters.qnr'

    _flog = None
    _adrs = None

    @classmethod
    def globals(cls):
        return globals()

    def __init__(self, specs=(), simple=None, **kw):
        super().__init__(**kw)
        self.extend(specs or config.exclude_specs)
        self.simple = simple or AdrFilter()

    def __repr__(self):
        s = '{}('.format(type(self).__name__)
        s += '{!r}, '.format(tuple(sorted(self.keys())))
        s += '{})'.format(pp.pformat(self.simple, indent=4))
        return s

    @property
    def flog(self):
        if self._flog is None:
            self._flog = Flog.create(self.base, self.realm)
            self._flog.clear()
        return self._flog

    @property
    def adrs(self):
        if self._adrs is None:
            self._adrs = FAdrs.create(self.base, self.realm)
            self._adrs.clear()
        return self._adrs

    def extend(self, specs):
        for s in specs:
            s = s.lower()
            self[s] = RAdrFilter(s)

    def probe(self, adr):
        r = self.simple.probe(adr)
        if r is None:
            for f in self.values():
                r = f.probe(adr)
                if r is not None:
                    return r
        else:
            return r
        self.adrs.incr(adr)

    def save(self, pref=None):
        super().save(pref)
        if self._flog:
            self._flog.save(pref)
        if self._adrs:
            self._adrs.save(pref)


class Flog(Resource):

    _res_path = config.qnar_dst + 'filts/flog.qnr'

    @classmethod
    def globals(cls):
        return globals()

    def __repr__(self):
        es = pp.pformat(self._elems, indent=4)
        return '{}({})'.format(type(self).__name__, es)

    def append(self, cur, fields):
        ls = self.setdefault(cur, [])
        ls.append(fields)


class FAdrs(Resource):

    _res_path = config.qnar_dst + 'filts/fadrs.qnr'

    @classmethod
    def globals(cls):
        return globals()

    def __repr__(self):
        es = pp.pformat(self._elems, indent=4)
        return '{}({})'.format(type(self).__name__, es)

    def incr(self, adr):
        self[adr] = self.setdefault(adr, 0) + 1

    def splits(self):
        ls = {}
        ds = {}
        for a, c in self.items():
            l, d = a.split('@')
            fs = d.split('.')
            d = fs[-2] + '.' + fs[-1]
            ls[l] = ls.setdefault(l, 0) + c
            ds[d] = ds.setdefault(d, 0) + c
        return ds, ls


if __name__ == '__main__':
    from .args import MArgs
    from .resource import resource

    a = MArgs()
    a = a.parse_args()
    with resource(FAdrs.create(a.base, a.files[0])) as fa:
        ds, ls = fa.splits()
        for d, n in sorted(ds.items(), key=lambda x: x[0], reverse=False):
            print("'" + d + "',")
        for l, n in sorted(ls.items(), key=lambda x: x[0], reverse=False):
            print("'" + l + "',")
