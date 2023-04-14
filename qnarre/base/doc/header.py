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

from .date import Date
from .log import Logger
from .nominals import flags
from .exporter import Exporter
from .category import Subjects
from .sanitizer import sanitize
from .error import ExcludeException
from .base import config, Adrs, Hdr, traits_for
from .base import LnkDate, LnkSubject, LnkFrom, LnkTo, LnkCc, LnkBcc

log = Logger(__name__)

on_wrote = r'^\W*on .+?wrote:$'
on_wrote = re.compile(flags + on_wrote)


class Line:

    keys = {
        'date: ': LnkDate.label,
        'sent: ': LnkDate.label,
        'subject: ': LnkSubject.label,
        'from: ': LnkFrom.label,
        'to: ': LnkTo.label,
        'cc: ': LnkCc.label,
        'bcc: ': LnkBcc.label,
        'in-reply-to: ': None,
        'reply-to: ': None
    }
    has_adrs = 'has_adrs'
    has_date = 'has_date'
    ignore = 'ignore'

    key = None

    def __init__(self, txt, **_):
        self.txt = t = txt
        if t:
            colon = ': '
            ps = t.split(colon, 1)
            if len(ps) == 2:
                try:
                    v = self.keys[ps[0].lower() + colon]
                    self.key = self.ignore if v is None else v
                except KeyError:
                    return
                    if ' ' not in ps[0]:
                        log.warning('Key {} not recognized', ps[0])
            elif on_wrote.match(t):
                if t.startswith('On '):
                    self.key = LnkFrom.label
            elif Adrs.has_adr(t):
                self.key = self.has_adrs
            elif Date.has_date(t):
                self.key = self.has_date


def extract(dst, src):
    for f in Hdr._fields:
        try:
            v = src[f]
            if v:
                setattr(dst, f, v)
            else:
                delattr(dst, f)
        except:
            pass
    return dst


def merge(dst, src):
    for f in Hdr._fields:
        o = getattr(dst, f, ())
        if isinstance(o, tuple):
            n = getattr(src, f, ())
            if n and isinstance(n, tuple):
                n = tuple(sorted(set((*o, *n))))
                if n != o:
                    setattr(dst, f, n)
    return dst


class TxtFields:
    @classmethod
    def extract(cls, raw, **_):
        date, from_, to, host, txt = raw
        to = (to, from_, None if from_ == host else host)
        from_ = (from_, None, None)
        return extract(cls(), locals()), txt


class ScrFields:
    @classmethod
    def extract(cls, raw, **_):
        date, topic, from_, txt = raw
        from_ = (from_, None, None)
        return extract(cls(), locals()), txt


class InlFields:

    on, wrote = 'On ', ' wrote:'

    @classmethod
    def extract(cls, quote, *, ctxt, **_):
        h = quote[0]
        if not h.startswith(cls.on) or not h.endswith(cls.wrote):
            raise ValueError('Not an inline message')
        txt = '\n'.join(quote[1:]).strip()
        try:
            h = h[len(cls.on):-len(cls.wrote)]
            i = h.index(':') + 6
            try:
                date = Date.from_inl(h[:i])
            except:
                i -= 3
                date = Date.from_inl(h[:i])
            from_ = Adrs.from_txt(h[i + 1:])
            return extract(cls(), locals()), txt
        except Exception as e:
            if txt and txt not in ctxt.nominals:
                raise e


class FwdFields:
    @classmethod
    def extract(cls, quote, *, ctxt, **_):
        date = from_ = prev = exc = None
        txt = ''
        for i, ln in enumerate(quote):
            if not ln:
                txt = '\n'.join(quote[i + 1:]).strip()
                break
            ps = ln.split(':', 1)
            try:
                if len(ps) == 2:
                    k, v = ps
                    k = k.lower()
                    v = v.strip()
                    prev = None
                elif prev:
                    k, p = prev
                    v = p + ' ' + ln
                else:
                    # print(quote)
                    raise ValueError('Forward fields need label: ' + ln)
                if k == 'date' or k == 'sent':
                    date = Date.from_fwd(v)
                elif k == 'subject':
                    subject = v
                    prev = k, v
                elif k == 'from':
                    from_ = Adrs.from_txt(v)
                elif k == 'to':
                    to = Adrs.from_txt(v)
                    prev = k, v
                elif k == 'cc':
                    cc = Adrs.from_txt(v)
                    prev = k, v
                elif k == 'bcc':
                    bcc = Adrs.from_txt(v)
                    prev = k, v
                elif k == 'in-reply-to':
                    replying = v
                elif k == 'message-id':
                    record_id = v
                else:
                    raise ValueError('Unrecognized forward label: ' + k)
            except Exception as e:
                exc = exc or e
        if not exc and date and from_:
            return extract(cls(), locals()), txt
        if txt and txt not in ctxt.nominals:
            if exc:
                raise exc
            raise ValueError('No date and/or sender for forward message')


class MixFields:
    @classmethod
    def extract(cls, form, **kw):
        txt = form.pop('txt', ())
        # for v in form.values():
        #    print(v)
        q = [l for l in (*form.values(), *txt)]
        ln = form.setdefault(LnkFrom.label, config.def_from)
        if ln.startswith('On '):
            return InlFields.extract(q, **kw)
        elif ln.startswith('From: '):
            return FwdFields.extract(q, **kw)
        log.warning('Form not recognized {!r}', form.items())
        raise ValueError('Form not recognized')


class DocFields:
    @classmethod
    def extract(cls, raw, *, ctxt, **kw):
        date, topic, ls = raw
        subject = date.short
        exc = None
        txt = ''
        if topic in ('orders', 'transcripts'):
            from_ = Adrs.from_txt('Court')
        if topic in ('affidavits', 'exhibits', 'hearings', 'trials'):
            to = Adrs.from_txt('Court')
        for i, l in enumerate(ls):
            if not l:
                txt = '\n'.join(ls[i + 1:]).strip()
                break
            ps = l.split(':', 1)
            try:
                if len(ps) == 2:
                    k, v = ps
                    k = k.lower()
                    v = v.strip()
                else:
                    # print(ls)
                    raise ValueError('Doc fields need label: ' + l)
                if k == 'title':
                    title = v
                elif k == 'subject':
                    subject = v
                elif k == 'summary':
                    summary = v
                elif k == 'from':
                    from_ = Adrs.from_txt(v)
                elif k == 'to':
                    to = Adrs.from_txt(v)
                elif k == 'cc':
                    cc = Adrs.from_txt(v)
                elif k == 'replying':
                    replying = v
                elif k == 'source':
                    source = v
                elif k == 'tags':
                    tags = (w.strip() for w in v.split(','))
                    tags = tuple(sorted(set(w for w in tags if w)))
                else:
                    raise ValueError('Unrecognized doc label: ' + k)
            except Exception as e:
                exc = exc or e
        if not exc and date and from_:
            if not txt:
                txt = summary
            return extract(cls(), locals()), txt
        if txt and txt not in ctxt.nominals:
            if exc:
                raise exc
            raise ValueError('No date and/or sender for doc')


class EmlFields:
    @classmethod
    def extract(cls, raw, **_):
        idx = None
        date = None

        def _value(f):
            nonlocal idx
            if f.endswith('_'):
                f = f[:-1]
            f = f.replace('_', '-')
            f = 'in-reply-to' if f == 'replying' else f
            v = raw.get_all(f)
            if isinstance(v, list):
                c = len(v)
                if c == 0:
                    v = None
                elif c == 1:
                    v = v[0]
                else:
                    for i in v[1:]:
                        if i and i != v[0]:
                            if f == 'message-id':
                                for i, s in enumerate(v):
                                    if s.endswith(r'@mx.google.com>'):
                                        idx = i
                                        return s
                            if idx is None:
                                log.info('Field {} with multi values {}', f, v)
                                v = v[0]
                            else:
                                v = v[idx]
                            break
                    else:
                        v = v[0]
            if v and f == 'date':
                v = v.datetime.replace(microsecond=0).astimezone()
                if not v.second:
                    v = v.replace(second=1)
                nonlocal date
                date = v = Date(v)
            return v

        fs = {f: _value(f) for f in Hdr._fields}
        return extract(cls(), fs), None


class Header(Exporter):
    def __init__(self, fields, ctxt=None, raise_exclude=True, **_):
        super().__init__()
        for k, v in fields.items():
            setattr(self, k, v)
        if ctxt:
            f = self.from_
            if hasattr(f, 'addresses'):
                pf, from_ = ctxt.slugs_for(f)
            else:
                pf, from_ = ctxt.slugs_for(*f)
            t = self.to
            if not t or hasattr(t, 'addresses'):
                pt, to = ctxt.slugs_for(t)
            else:
                pt, to = ctxt.slugs_for(*t)
            c = self.cc
            if not c or hasattr(c, 'addresses'):
                pc, cc = ctxt.slugs_for(c)
            else:
                pc, cc = ctxt.slugs_for(*c)
            b = self.bcc
            if not b or hasattr(b, 'addresses'):
                pb, bcc = ctxt.slugs_for(b)
            else:
                pb, bcc = ctxt.slugs_for(*b)
            # if not to and not cc and not bcc:
            # raise ExcludeException()
            ps = (pf, pt, pc, pb)
            if not any(ps) and any([True for p in ps if p is False]):
                if raise_exclude:
                    raise ExcludeException()
            s = self.subject
            try:
                s = sanitize(s) if s else ''
            except UnicodeError:
                if raise_exclude:
                    raise ExcludeException()
                s = 'UnicodeError'
            subject = Subjects.dejunk(s, ctxt)
            self.extract(locals())

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, vars(self))

    @property
    def name(self):
        return self.date.name

    @property
    def slug(self):
        return self.date.slug

    extract = extract
    merge = merge

    cmp_ks = set((LnkFrom.label, LnkTo.label, LnkCc.label, LnkBcc.label,
                  LnkSubject.label))

    def compare(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        dc = self.date.compare(other.date)
        if dc is config.EQ:
            ks = set((*vars(self).keys(), *vars(other).keys())) & self.cmp_ks
            for k in ks:
                try:
                    if getattr(self, k) != getattr(other, k):
                        break
                except AttributeError:
                    break
            else:
                return config.EQ
        elif dc in (config.LT, config.GT):
            k = LnkFrom.label
            try:
                if getattr(self, k) == getattr(other, k):
                    return dc
            except AttributeError:
                pass

    def mboxer(self, ctxt, **_):
        yield 'message-id', self.record_id or self.name
        if self.replying:
            yield 'in-reply-to', self.replying
        yield 'date', self.date.raw
        yield 'from', ctxt.name(self.from_[0])
        if self.to:
            yield 'to', ', '.join(ctxt.name(s) for s in self.to)
        if self.cc:
            yield 'cc', ', '.join(ctxt.name(s) for s in self.cc)
        if self.bcc:
            yield 'bcc', ', '.join(ctxt.name(s) for s in self.bcc)

    def plainer(self, ctxt, **_):
        f = ctxt.name(self.from_[0])
        t = self.to or ()
        cc = self.cc or ()
        if len(t) + len(cc) > 1:
            t = ' to audience'
        elif t and ctxt[config.DEFAULT].name not in (*f, *t):
            t = ' to ' + t[0]
        else:
            t = ''
        yield '**On {}, {} wrote{}:**'.format(self.date.to_inl, f, t)
        yield '\n'

    def htmer(self, just, frame, ctxt, **_):
        ts = [traits_for(f) for f in self.from_]
        if just:
            j = just.calc_just(t.justify for t in ts)
        else:
            j = 'justify-content-start'
        bs = (t.background for t in ts if t.background is not None)
        yield frame[2].format(j, next(bs, 'e8e8e8'))
        if just:
            yield frame[3].format(self.date.to_inl, ctxt.name(self.from_[0]))
        yield '\n'

    def blogger(self, **_):
        v = self.title
        yield v
        yield '=' * len(v)
        yield ':Date: ' + self.date.to_rst
        yield ':From: ' + ', '.join(self.from_)
        v = self.to
        if v:
            yield ':To: ' + ', '.join(v)
        v = self.cc
        if v:
            yield ':Cc: ' + ', '.join(v)
        v = self.bcc
        if v:
            yield ':Bcc: ' + ', '.join(v)
        yield '\n'

    def footer(self, **_):
        pass


for f in Hdr._fields:
    setattr(Header, f, None)
