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
import os

import pathlib as pth
import collections as co

from .date import Date
from .log import Logger
from .header import Line
from .error import ExcludeException
from .sanitizer import QNERR, sanitize
from .base import config, LnkDate, LnkFrom

log = Logger(__name__)


def scanner(path, suffs, files=(), pfix=None, **_):
    n = 0
    files = files or ('test', )
    every = 'all' in files

    def scan_dir(path):
        with os.scandir(path) as es:
            for e in es:
                p = pth.Path(e.path)
                if not p.name.startswith('.'):
                    ss = ''.join(p.suffixes)
                    if p.is_file() and ss in suffs:
                        if not pfix or p.stem.endswith(pfix):
                            st = p.name.replace(ss, '')
                            if every or st in files:
                                nonlocal n
                                n += 1
                                yield p
                    elif p.is_dir():
                        yield from scan_dir(p)

    yield from scan_dir(path)
    if every:
        log.info('{} has {} files', path, n)


def liner(path, clip=None, **kw):
    s = path.suffix
    if s == '.pdf':
        if clip is None:
            clip = 7
            if b'PhoneView' not in path.read_bytes():
                clip = 2
        c = config.pdf_context()
        g = config.pdf_generator(c, **kw)
        e = config.pdf_executor(c, g)
        with open(path, "rb") as f:
            for p in config.pdf_page.get_pages(f):
                ls = e.process_page(p)
                for l in ([l for l in ls][:-clip] if clip else ls):
                    yield sanitize(' '.join(l.split()))
        g.close()
    elif s == '.txt':
        # print(str(path))
        # sanitize(path)
        with open(path, encoding='ascii', errors=QNERR) as f:
            for l in f:
                yield ' '.join(l.split())
    elif s == '.md':
        # print(str(path))
        # sanitize(path)
        with open(path, encoding='ascii', errors=QNERR) as f:
            for l in f:
                yield ' '.join(l.split())


def msger(path, src=None, msg_range=None, **kw):
    if src is None:
        from .mboxes import Mbox
        src = Mbox.qsrc
    n = 0
    for m in src(path, **kw):
        if not msg_range or n in msg_range:
            yield m
        n += 1
        if msg_range and n > max(msg_range):
            break
    if not msg_range:
        log.info('{} has {} messages', path.name, n)


ws = re.compile(r'[a-zA-Z_]+,?', re.ASCII)


def names(txt='', default='Cyndi'):
    t = ' '.join(ws.findall(txt))
    t = t or default
    return ', '.join(t.split(','))


class Reader:
    def __init__(self, path):
        self.path = path

    def pdf_to_txt(self, **kw):
        for p in scanner(self.path, ('.pdf', ), **kw):
            p2 = p.with_suffix('.txt')
            if p2.exists():
                log.warning('File {} exists, skipped', str(p2))
            else:
                p2.write_text('\n'.join(liner(p, **kw)), 'ascii', QNERR)

    def from_tbox(self, *, ctxt, cntr, **kw):
        on = ' on '
        sent = 'Sent '
        recv = 'Received '
        recv_from = 'Received from '

        def src(path, **_):
            title = date = txt = None
            from_ = host = config.DEFAULT
            to = names()
            for ln in liner(path, **kw):
                if title and ln == title:
                    continue
                elif ln.startswith('Messages with'):
                    if date:
                        t = '' if txt is None else txt
                        yield date, from_, to, host, t
                    title, date, txt = ln, None, None
                    from_ = host = config.DEFAULT
                    to = names(ln[len('Messages with'):])
                    continue
                elif ln.startswith('Messages between'):
                    if date:
                        t = '' if txt is None else txt
                        yield date, from_, to, host, t
                    title, date, txt = ln, None, None
                    from_ = host = names(ln[len('Messages between'):])
                    to = ', '.join((from_, names()))
                    continue
                elif ln.startswith('Messages'):
                    if date:
                        t = '' if txt is None else txt
                        yield date, from_, to, host, t
                    title, date, txt = ln, None, None
                    from_ = host = config.DEFAULT
                    to = names()
                    continue
                if ln.startswith(sent) or ln.startswith('Send To '):
                    if ln.startswith(sent):
                        i = ln.find(on)
                        i = (i + len(on)) if i > 0 else len(sent)
                        ln = ln[i:]
                        i = ln.rfind(':')
                        i = ln.rfind('!') if i < 0 else i
                        ln = ln if i < 0 else ln[:(i + 6)]
                    else:
                        i = ln.find(' at ')
                        ln = ln[(i + len(' at ')):]
                    try:
                        d = Date.from_txt(ln)
                    except ValueError as e:
                        log.info('Failed to extract date {}', e)
                    else:
                        if date:
                            t = '' if txt is None else txt
                            yield date, from_, to, host, t
                        date = d
                        from_ = host
                        txt = None
                        continue
                elif ln.startswith(recv) or ln.startswith('From '):
                    i = ln.find(on)
                    if ln.startswith(recv_from):
                        f = names(ln[len(recv_from):i], to)
                        ln = ln[(i + len(on)):]
                    elif ln.startswith('From '):
                        i = ln.find(' at ')
                        f = names(ln[len('From '):i], to)
                        ln = ln[(i + len(' at ')):]
                    else:
                        i = (i + len(on)) if i > 0 else len(recv)
                        ln = ln[i:]
                        i = ln.rfind(':')
                        i = ln.rfind('!') if i < 0 else i
                        if i >= 0:
                            i = i + 6
                            f = names(ln[i:], to)
                            ln = ln[:i]
                    try:
                        d = Date.from_txt(ln)
                    except ValueError as e:
                        log.info('Failed to extract date {}', e)
                    else:
                        if date:
                            t = '' if txt is None else txt
                            yield date, from_, to, host, t
                        date = d
                        from_ = f
                        txt = None
                        continue
                if txt is None:
                    txt = ln
                else:
                    txt = '\n'.join((txt, ln))
            if date:
                t = '' if txt is None else txt
                yield date, from_, to, host, t

        for p in scanner(self.path, ('.txt'), **kw):
            ctxt.current = n = str(p.relative_to(self.path))
            cntr.retitle(n)
            for m in msger(p, src, **kw):
                yield n, m

    def from_sbox(self, *, ctxt, cntr, **kw):
        def src(path, date=None, topic=None, **kw):
            from_ = txt = None
            for l in liner(path, **kw):
                ps = l.split('::')
                if len(ps) == 2:
                    if from_:
                        date = date.next_sec()
                        t = '' if txt is None else txt
                        # print(date, from_, repr(t))
                        yield date, topic, from_, t
                    from_, txt = ps
                else:
                    if txt is None:
                        txt = l
                    else:
                        txt = '\n'.join((txt, l))
            if from_:
                date = date.next_sec()
                t = '' if txt is None else txt
                # print(date, from_, t)
                yield date, topic, from_, t

        p = self.path
        ctxt.current = t = p.name
        cntr.retitle(t)
        for d, p, _, i in Date.scanner(p, suffs=('.txt')):
            d = Date(d.raw).next_hour(i * 3)
            for m in msger(p, src, **kw, date=d, topic=t):
                yield str(p.relative_to(self.path)), m

    def from_mbox(self, *, ctxt, cntr, **kw):
        es = co.OrderedDict()
        us = co.OrderedDict()
        for p in scanner(self.path, (
                '.mbox',
                '.mbox.xz',
        ), **kw):
            ctxt.current = n = p.stem
            cntr.retitle(p.name)
            for m in msger(p, **kw):
                if 'Drafts' in m.get('X-Gmail-Labels', ()):
                    cntr.incr('-')
                    continue
                mid = m['message-id']
                try:
                    if ctxt.mids[mid] is config.EXCLUDED:
                        cntr.incr('-')
                        continue
                except KeyError:
                    us[p] = 1 + us.setdefault(p, 0)
                try:
                    yield n, m
                except ExcludeException:
                    es[p] = 1 + es.setdefault(p, 0)
        for p, u in us.items():
            e = es.get(p, 0)
            log.info('{} has {} unique and {} excluded messages', p.name, u, e)

    def from_bbox(self, *, ctxt, cntr, **kw):
        date = LnkDate.label
        from_ = LnkFrom.label

        def src(path, **_):
            form = {}
            prev = None
            for ln in liner(path, **kw):
                ln = Line(ln)
                if ln.key is ln.ignore:
                    continue
                if ln.key in form:
                    yield form
                    form = {}
                    prev = None
                t = ln.txt
                if ln.key:
                    if ln.key is ln.has_adrs:
                        if prev and prev.key:
                            form[prev.key] += ' ' + t
                        else:
                            if from_ in form:
                                yield form
                                form = {}
                                prev = None
                            form[from_] = 'From: ' + t
                        continue
                    elif ln.key is ln.has_date:
                        if date in form:
                            for n in config.book_names:
                                if t.startswith(n):
                                    yield form
                                    form = {}
                                    f = 'On ' + t[len(n):] + ', '
                                    f += n + ' wrote:'
                                    form[from_] = f
                                    prev = None
                                    break
                            else:
                                log.warning('Already dated {}, new one {}',
                                            form[date], ln.txt)
                            continue
                        form[date] = 'Date: ' + t
                    else:
                        form[ln.key] = t
                        prev = ln
                        continue
                else:
                    form.setdefault('txt', []).append(t)
                prev = None
            yield form

        for p in scanner(self.path, ('.txt', ), **kw):
            ctxt.current = n = p.stem
            cntr.retitle(p.name)
            for m in msger(p, src, **kw):
                yield n, m

    def from_docs(self, *, ctxt, cntr, stamp=True, **kw):
        def src(path, date=None, topic=None, **_):
            yield date, topic, tuple(liner(path, **kw))

        with os.scandir(self.path) as es:
            c = 0
            for e in es:
                p = pth.Path(e.path)
                if p.is_dir():
                    ctxt.current = t = p.name
                    cntr.retitle(t)
                    for d, p, _, i in Date.scanner(p, suffs=('.md')):
                        d = Date(d.raw)
                        d.micro = c * 100 + i
                        for m in msger(p, src, **kw, date=d, topic=t):
                            yield str(p.relative_to(self.path)), m
                    if stamp:
                        c += 1

    def from_main(self, **kw):
        yield from self.from_docs(**kw, stamp=False)


if __name__ == '__main__':
    from .args import BArgs
    a = BArgs()
    a.add_argument('files', nargs='*', help='Files to read')
    a.add_argument('-c', '--clip', help='Lines to clip')
    a = a.parse_args()
    Reader(a.base).pdf_to_txt(**a.kw)
