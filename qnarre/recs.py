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

# from difflib import unified_diff

from .log import Logger
from .base import config
from .edit import redact
from .chain import Chains
from .counter import counters
from .resource import Resource
from .record import EmlRec, TxtRec, MixRec, ScrRec, DocRec, PicRec
from .record import StoryRec, BlogRec

from .date import Date  # needed dynamically
from .record import InlRec, FwdRec
from .header import Header

log = Logger(__name__)

src_to_cls = {
    config.MBOX: EmlRec,
    config.TBOX: TxtRec,
    config.BBOX: MixRec,
    config.SBOX: ScrRec,
    config.REPO[1:-1]: DocRec,
    config.BLOG[1:-1]: BlogRec,
    config.MAIN[1:-1]: StoryRec,
    config.PICS: PicRec,
}


class Recs(Resource):

    _res_path = config.qnar_dst + 'recs.qnr'

    _chains = None

    @classmethod
    def globals(cls):
        return globals()

    @property
    def recs(self):
        return self.elems

    @property
    def chains(self):
        if self._chains is None:
            self._chains = Chains.create(self.base, self.realm)
        return self._chains

    def rename_msg(self, old, new):
        super().rename(old, new)
        for m in self.values():
            m.rename(old, new)

    def save(self, pref=None):
        super().save(pref)
        if self._chains:
            self._chains.save(pref)

    def no_secs(self):
        ns = {}
        for r in self.values():
            for d in r.zero_secs:
                if d in ns:
                    ns[d].append(r)
                else:
                    ns[d] = [r]
        return ns

    def importer(self, rec, *, no_secs, only_one=None, **kw):
        if only_one is None or rec.hdr.name.startswith(only_one):
            qs = []
            for lv, c in rec.reducer(**kw):
                if c:
                    for m in self.importer(c, **kw, no_secs=no_secs):
                        yield m
                        if lv == 1 and c is m:
                            qs.append(c.name)
            if qs:
                rec.hdr.quoting = tuple(qs)
            rss = []
            for k in rec.zero_secs:
                try:
                    rs = no_secs[k]
                except KeyError:
                    no_secs[k] = rs = []
                else:
                    rec = rec.consolidate(rs, **kw)
                    if not rec:
                        return
                rss.append(rs)
            rec.register(**kw)
            for rs in rss:
                rs.append(rec)
            yield rec

    import_args = ((('scanned', '.'), ('excluded', '-'), ('imported', '+'),
                    ('equal', '='), ('less', '<'), ('greater',
                                                    '>'), ('failed', 'F')), '')

    def import_from(self, src, **kw):
        kw.update(no_secs=self.no_secs())  # , only_one='10-10-25|22:42:44')
        with counters(self.import_args, kw) as cs:
            src = self.base / src
            for c in src_to_cls[src.stem].importer(src, **kw):
                for r in self.importer(c, **kw):
                    self[r.name] = r
                    cs.incr('+')
            for r in self.values():
                r.rectify(**kw)
            return cs

    def copy_from(self, src, editor, **kw):
        with counters(self.import_args, kw) as cs:
            for r in src.recs.values():
                n = r.name
                if n not in self:
                    h = Header(vars(r.hdr), **kw)
                    h.subject = redact(h.subject)
                    r = type(r)(h, r.source, editor(r.text(src)))
                    r.register(**kw)
                    self[n] = r
                    cs.incr('+')
                else:
                    cs.incr('.')
            return cs

    def grapher(self, ctxt, types=(), **kw):
        for r in self.values():
            c = type(r)
            if types is not None and (not types or c in types):
                yield r.name, r.text(ctxt), c
            yield from r.edger(**kw)
        for a in (getattr(ctxt, n) for n in ('topics', 'subjects', 'sources')):
            yield from a.grapher(**kw)

    def chainer(self, cntr, **kw):
        ts = {}
        for r in self.chains.chainer(self.grapher(**kw), **kw):
            ts.setdefault(r.topic(**kw), []).append(r)
            cntr.incr('.')
        yield from ts.items()
