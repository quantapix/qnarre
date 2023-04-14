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

import pprint as pp

from .log import Logger
from .base import config
from .header import Header
from .counter import counters
from .connect import Connects
from .exporter import Exporter
from .resource import Resource
from .justifier import Justifier
from .base import Record, LnkQuoting, LnkReplying, traits_for
from .base import LnkTopic, LnkSubject, LnkProximity, LnkAudience, LnkSource

from .date import Date  # needed for repr reading

log = Logger(__name__)


class ChnHeader(Header, Justifier):
    def __init__(self, hdr):
        super().__init__(vars(hdr))


class Chain(Exporter):
    def __init__(self, names):
        self.names = tuple(sorted(set(names)))

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.names)

    @property
    def name(self):
        return 'Chain ' + self.names[-1]

    def topic(self, **_):
        return self._topic

    def subject(self, **_):
        return self._subject

    def collapse(self, ctxt):
        r = ctxt.recs[self.names[-1]]
        h = r.hdr
        src = h.source or r.source or ''
        self.hdr = hdr = ChnHeader(h)
        ps = set()
        tp = sb = ''
        f = hdr.from_
        for n in self.names[:-1]:
            r = ctxt.recs[n]
            h = r.hdr
            s = h.source or r.source or ''
            if src and s and src != s:
                src = None
            src = None if src is None else (src or s)
            hdr.merge(h)
            rt, rs = r.topic(ctxt), r.subject(ctxt)
            assert not tp or not rt or tp == rt
            tp = rt or tp
            assert not sb or not rs or sb == rs
            sb = rs or sb
            ps.add(h.date.proximity)
        hdr.init_justs(traits_for(f).justify for f in hdr.from_)
        hdr.from_ = f
        self._topic = tp
        if not sb:
            if len(ps) == 1:
                hdr.title = 'On {}'.format(hdr.date.short)
                sb = src.split('.')[0] if src else hdr.title
            else:
                sb = 'Chained'
        self._subject = sb

    def plainer(self, ctxt, **kw):
        for m in self.names:
            r = ctxt.recs[m]
            yield from r.hdr.plainer(ctxt, **kw)
            yield from r.plainer(**kw, ctxt=ctxt)
            yield '\n'

    def htmer(self, frame=None, ctxt=None, **kw):
        yield frame[0]
        yield frame[1]
        for m in self.names:
            r = ctxt.recs[m]
            yield from r.hdr.htmer(self.hdr, frame, ctxt, **kw)
            yield from r.htmer(**kw, ctxt=ctxt)
            yield frame[-3]
        yield frame[-2]
        yield frame[-1]

    def blogger(self, **kw):
        yield from self.hdr.blogger(**kw)
        yield '\n'.join(self.plainer(**kw))
        yield from self.hdr.footer(**kw)


fields = (Record, LnkQuoting, LnkReplying, LnkTopic, LnkSubject, LnkProximity,
          LnkAudience, LnkSource)
fields = tuple((f.label, '') for f in fields)


class Chains(Resource):

    _res_path = config.qnar_dst + 'chains.qnr'

    _graphs = None

    @classmethod
    def globals(cls):
        return globals()

    def __init__(self, data=None, **kw):
        self._seed, self._adjs, elems = data or ((), (), ())
        elems = {c.name: c for c in elems}
        super().__init__(elems, **kw)

    def __repr__(self):
        es = tuple(sorted(self.values(), key=lambda c: c.name))
        es = pp.pformat(es, indent=4)
        return '{}(({!r}, {!r}, {}))'.format(
            type(self).__name__, self._seed, self._adjs, es)

    @property
    def graphs(self):
        if self._graphs is None:
            self._graphs = Connects(self._seed)
        return self._graphs

    graph_args = (fields, 'Graphing:', '')
    collapse_args = ((('purged', 'd'), ('quoting', 'q'), ('replying', 'r'),
                      ('subject', 's'), ('proximity', 'p'), ('audience', 'a'),
                      ('failed', 'F')), 'Collapsing:', '')

    def chainer(self, src, ctxt, **kw):
        gs = self.graphs
        with counters(self.graph_args, kw) as cs:
            cs.retitle()
            gs.grow_from(src, self._adjs, **kw)
        del kw['cntr']
        with counters(self.collapse_args, kw) as cs:
            cs.retitle()
            gs.collapse_all(**kw)
        rg = gs.record
        for r in sorted(rg.nodes()):
            try:
                c = Chain((r, *rg.node[r][config.CHAIN]))
            except KeyError:
                yield ctxt.recs[r]
            else:
                c.collapse(ctxt)
                self[c.name] = c
                yield c
