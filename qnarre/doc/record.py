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

from .junk import Junk
from .log import Logger
from .reader import Reader
from .exporter import Exporter
from .sanitizer import sanitize
from .error import ExcludeException
from .header import DocFields, EmlFields, Header
from .base import config, Hdr, Record, LnkProximity, LnkAudience
from .nominals import compare, para_make, para_split, para_join, quoter
from .header import TxtFields, ScrFields, InlFields, FwdFields, MixFields

log = Logger(__name__)


class Record(Record, Exporter):

    raw = None
    junk = None
    source = None
    no_date = False

    label = 'record'

    _text = None

    @classmethod
    def filterer(cls, path, ctxt, cntr, **kw):
        kw.update(ctxt=ctxt, cntr=cntr)
        rs = iter(cls.reader(Reader(path), **kw))
        r = False
        while True:
            r = r or next(rs)
            try:
                src, raw = r
                fs = cls.fields.extract(raw, **kw)
                if fs:
                    fs, txt = fs
                    yield raw, Header(vars(fs), **kw), txt, src
            except ExcludeException as e:
                ctxt.filters.flog.append(ctxt.current, vars(fs))
                cntr.incr('-')
                r = rs.throw(e)
                continue
            except ValueError as e2:
                log.warning('Failed reading record {}', e2)
                # assert False
                cntr.incr('F')
            r = False

    @classmethod
    def importer(cls, path, ctxt, **kw):
        kw.update(ctxt=ctxt, sort_mbox=True)
        for raw, hdr, txt, src in cls.filterer(path, **kw):
            if txt is None:
                txt = '\n'.join(ctxt.extract(hdr.record_id, raw, **kw))
            if 'UnicodeError' not in txt:
                yield cls(hdr, src, txt)

    @classmethod
    def create_from(cls, src, quote, **kw):
        fs = cls.fields.extract(quote, **kw)
        if fs:
            fs, txt = fs
            return cls(Header(vars(fs), **kw), src, txt)

    def __init__(self, hdr, source=None, raw=None):
        self.hdr = hdr
        if source is not None:
            self.source = source
        if raw is not None:
            self.raw = raw

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.hdr == other.hdr
        return NotImplemented

    def __repr__(self):
        s = self.source
        if s:
            return '{}({!r}, {!r})'.format(type(self).__name__, self.hdr, s)
        return '{}({!r})'.format(type(self).__name__, self.hdr)

    @property
    def name(self):
        return self.hdr.name

    @property
    def slug(self):
        return self.hdr.slug

    @property
    def audience(self):
        ls = (getattr(self.hdr, f, ()) for f in ('from_', 'to', 'cc', 'bcc'))
        ns = sorted(set(e for s in ls if s is not None for e in s))
        return ', '.join(ns)

    @property
    def zero_secs(self):
        return self.hdr.name,

    def text(self, ctxt=None, **_):
        if self._text is None:
            r = self.raw
            if r is None:
                self._text = para_join(ctxt.texts.get(self.name, ()))
            else:
                if self.junk:
                    r = sanitize(r)
                    self.raw = self.junk.dejunk_text(r)
                    self.junk = False
                return self.raw
        elif self.raw is not None:
            del self.raw
        return self._text

    def topic(self, ctxt=None, **_):
        if self._topic is None:
            t = ctxt.topics.resolve_all(self.name, self.subject(ctxt),
                                        self.audience, self.hdr.topic)
            self._topic = t or config.TBD
        return self._topic

    def subject(self, ctxt=None, **_):
        if self._subject is None:
            ss = ctxt.subjects
            self._subject = ss.resolve_all(self.name, self.hdr.subject)
        return self._subject

    def reducer(self, **kw):
        ts = []
        for lv, qs in quoter(self.text().splitlines()):
            if lv == 0:
                qs = '\n'.join(qs).strip()
                if qs:
                    ts.append(qs)
            else:
                es = []
                for m in (InlRec.create_from, FwdRec.create_from):
                    try:
                        yield lv, m(self.source, qs, **kw, raise_exclude=False)
                        break
                    except Exception as e:
                        es.append(e)
                        # import traceback as tb
                        # tb.print_tb(e.__traceback__)
                else:
                    f = 'Quoting failed {}\n{!r}\n{!r}'
                    log.warning(f, self.name, qs, es)
                    assert False
        self._text = para_make('\n'.join(ts))

    def expand(self, txt, ctxt):
        self._text = txt
        ctxt.texts.expand(self.hdr.name, para_split(txt))
        ctxt.nominals.append(txt)

    def consolidate(self, others, ctxt, cntr, **kw):
        ds = []
        old = None
        h = self.hdr
        t = self.text()
        for o in sorted(others, key=lambda m: m.name):
            oh = o.hdr
            hc = h.compare(oh)
            ot = o.text(ctxt)
            tc = compare(t, ot)
            if tc:
                if hc in (config.EQ, config.LT):
                    assert not old
                    if tc is config.GT:
                        # o.expand(t, ctxt)
                        pass
                    cntr.incr('=' if hc is config.EQ else '<')
                    return
                elif hc is config.GT:
                    if tc is config.GT:
                        # o.expand(t, ctxt)
                        pass
                    elif tc is config.LT:
                        t = ot
                    assert not old
                    old, o.hdr = oh.name, h
                    if self.source is not None:
                        o.source = self.source
                    self = o
                    continue
            ds.append(oh.date)
            if h.name == oh.name:
                h.date.after(ds)
        if not old:
            return self
        ctxt.rename_msg(old, self.name)
        self.register(ctxt)
        cntr.incr('>')

    def rename(self, old, new):
        h = self.hdr
        if h.replying == old:
            h.replying = new
        q = h.quoting
        if q and old in q:
            q = tuple(new if e == old else e for e in q)
            if q:
                h.quoting = q
            else:
                del h.quoting

    def rectify(self, ctxt, force=False, **_):
        h = self.hdr
        r = h.replying
        if r:
            try:
                h.replying = r = ctxt.mids[r]
            except KeyError:
                if not force:
                    return
        q = h.quoting
        if q and r in q:
            q = tuple(e for e in q if e != r)
            if q:
                h.quoting = q
            else:
                del h.quoting

    def register(self, ctxt, **_):
        h = self.hdr
        n, m = h.name, h.record_id
        if m:
            ctxt.mids[m] = n
            del h.record_id
        self.rectify(ctxt, force=True)
        s = self.subject(ctxt)
        if s:
            ctxt.subjects[s] = n
        t = self.topic(ctxt)
        if t:
            ctxt.topics[t] = n
        s = self.source
        if s:
            ctxt.sources[s] = n
        t = self.text()
        ctxt.texts.register(n, para_split(t))
        ctxt.nominals.append(t)

    def undirected(self, links=(), **_):
        if links is not None:
            h, n = self.hdr, self.name
            ls = (l for l in Hdr.links
                  if not l.directed and (not links or l in links))
            for l in ls:
                o = getattr(h, l.label)
                if isinstance(o, tuple):
                    for i in o:
                        if i:
                            yield i, n, l
                elif o:
                    yield o, n, l

    def edger(self, links=(), directed=True, **kw):
        if links is not None:
            h, n = self.hdr, self.name
            ls = (l for l in Hdr.links
                  if l.directed and (not links or l in links))
            for l in ls:
                o = getattr(h, l.label)
                if isinstance(o, tuple):
                    for i in o:
                        if i and '|' in i:
                            yield i, n, l
                elif o and '|' in o:
                    yield o, n, l
            a = LnkAudience
            if not links or a in links:
                yield self.name, self.audience, a
            if not directed:
                yield from self.undirected(links, **kw)


class TxtRec(Record):

    reader = Reader.from_tbox
    fields = TxtFields

    def edger(self, links=(), directed=True, **kw):
        if links is not None:
            yield from super().edger(links, directed, **kw)
            p = LnkProximity
            if not links or p in links:
                h = self.hdr
                yield h.name, h.date.proximity, p


class MixRec(Record):

    reader = Reader.from_bbox
    fields = MixFields
    junk = Junk()

    @property
    def zero_secs(self):
        return self.hdr.date.zero_secs


class ScrRec(TxtRec):

    no_date = True
    reader = Reader.from_sbox
    fields = ScrFields


class DocRec(Record):

    reader = Reader.from_docs
    fields = DocFields


class PicRec(DocRec):
    pass


class BlogRec(DocRec):
    pass


class StoryRec(DocRec):

    reader = Reader.from_main


class EmlRec(Record):

    reader = Reader.from_mbox
    fields = EmlFields
    junk = Junk()

    @property
    def zero_secs(self):
        return self.hdr.date.zero_secs


class InlRec(EmlRec):

    fields = InlFields
    junk = None


class FwdRec(EmlRec):

    fields = FwdFields
    junk = None
