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

import pathlib as pth

from .log import Logger
from .sanitizer import QNERR
from .base import config, Adrs
from .nominals import flags, nbs

log = Logger(__name__)


def splicer(txt):
    p = ''
    for l in txt.splitlines():
        if len(l) == 78 and l[-1] == '=':
            p += l[:-1]
        else:
            p += l
            yield ' '.join(p.split())
            p = ''


line = r'.+?\n'
lines = r'(.+?\n)*?'
line_2 = r'.+?==\n'

eml_junk = (
    r'[*]{4}',
    r' (GMT-05:00)',
    r'\[LINK:[^][]+\]',
    r'^X-Mailer: ' + line,
    r'^References: ' + line,
    r'^Return-Path: ' + line,
    r'^Mime-Version: ' + line,
    r'^Content-Type: ' + line,
    r'^From MAILER-DAEMON ' + line,
    r'^X-Gm-Message-State: ' + lines + line_2,
    r'^(X-Google-)?DKIM-Signature: ' + lines + line_2,
    r'^(X-)?Received: ' + lines + r'.+?\(P[D|S]T\)\n',
)

eml_junk = tuple(re.compile(flags + p) for p in eml_junk)

qt = r'(?P<qt>[> ]+)'

dig = r'[\d -]'
tel = r'[%.\d -]'

nb2 = r'[^][\n]'
ads = r'(?P<tx>([;, ]*' + Adrs.adr_pat + r'[;, ]*)+)'

split_junk = (
    (r'<tel:' + tel + r'+?>', '',
     r'(?P<lf><tel:' + tel + r'*?)\n' + qt + r'?(?P<rt>' + tel + r'+>)'),
    (r'<' + dig + r'+?>', '',
     r'(?P<lf><\d' + dig + r'*?)\n' + qt + r'?(?P<rt>' + dig + r'+>)'),
    (r'<mailto:' + nbs + r'+?>', '',
     r'(?P<lf><mailto:' + nbs + r'*?)\n' + qt + r'?(?P<rt>' + nbs + r'+>)'),
    (r'\[[ ]*mailto:(?P<tx>' + nb2 + r'+?)\]', r' \g<tx> ',
     r'(?P<lf>\[mailto:' + nb2 + r'*?)\n' + qt + r'?(?P<rt>' + nb2 + r'+\])'),
    (r'<' + ads + r'>', r' \g<tx> ',
     r'^(?P<lf>(?P<qt>>+ )?.*?<' + nbs + r'*)\n(?P=qt)(?P<rt>' + nbs + r'*>)'),
    (r'<(blocked::)?\W*(http:|https:|javascript:)' + nbs + r'+>', '',
     r'(?P<lf><http:' + nbs + r'*?)\n' + qt + r'?(?P<rt>' + nbs + r'+>)'),
    (r'<' + nbs +
     r'+?(.pdf|.jpg|.jpeg|.png|.gif|.tif|.doc|.mov|.docx|.ptx|.zip)>', '', ''),
    (r'\[ ?(cid|image|Description|http):' + nb2 + r'+?\]', '',
     r'(?P<lf>\[ ?(cid|image|Description|http):' + nb2 + r'*?)\n' + qt +
     r'?(?P<rt>' + nb2 + r'+\])'),
)

split_junk = tuple((re.compile(flags + p), r, re.compile(flags + s))
                   for p, r, s in split_junk)

ow = r'(On (?:(?!wrote:).)*\n(?:(?!On ).)*wrote:)$'
ow = re.compile(flags + ow)


def ow_splicer(txt):
    for e in ow.split(txt):
        if ow.match(e):
            yield e.replace('\n', ('' if e.endswith(' wrote:') else ' '))
        else:
            yield e


def defragment(txt):
    t = txt.strip()
    done = False
    while not done:
        done = True
        t = ''.join(ow_splicer(t))
        for p, r, s in split_junk:
            while p.pattern != flags:
                t, n = p.subn(r, t)
                if n:
                    done = False
                else:
                    break
            if s.pattern != flags:
                t, n = s.subn(r'\g<lf>\g<rt>', t)
                if n:
                    done = False
    return t


def simple_replacer(txt):
    for ln in txt.strip().splitlines():
        for p in config.line_junk:
            ln = ln.replace(p, '')
        for p, s in config.line_replace:
            ln = ln.replace(p, s)
        yield ln


def patch(txt):
    t = txt
    for p, r in tuple((re.compile(flags + p), r) for p, r in config.fixups):
        t = p.sub(r, t)
    for r in tuple(re.compile(flags + p) for p in config.quotes):
        t = r.sub(r'\g<qt> | \g<tx>', t)
    return t


nw = r'\W+'
nwc = re.compile(flags + nw)


def re_junks(name, rexes=None):
    rs = {} if rexes is None else rexes
    p = pth.Path.cwd() / name
    if p.exists():
        t = '\n'.join(splicer(p.read_text('ascii', QNERR))).lower()
        for s in t.split('\n\n\n'):
            if s:
                ss = [s for s in nwc.split(s) if s]
                r = '\n' + ' '.join(ss) + '\n'
                if r not in rs:
                    e = flags + r'^\W*' + '\W*'.join(ss) + '\W*$'
                    rs[r] = re.compile(e)
    else:
        log.warning('Defaults for junk were not found')
    return rs


class Junk:

    default = 'def_junks.txt'

    js = ()
    rejs = re_junks(default)

    _sorted_rejs = None

    @classmethod
    def junks_from(cls, path):
        rs = re_junks(path, cls.rejs)
        t = '\n'.join(sorted(rs.keys()))
        pth.Path(cls.default).write_text(t, 'ascii', QNERR)

    def __init__(self, junks=None):
        if junks is not None:
            self.js = junks

    @property
    def sorted_rejs(self):
        if self._sorted_rejs is None:
            ks = sorted(self.rejs.keys(), key=lambda k: len(k), reverse=True)
            self._sorted_rejs = tuple(self.rejs[k] for k in ks)
        return self._sorted_rejs

    def add(self, junks):
        self.js = *self.js, *junks

    def dejunk_line(self, line):
        ln = ' '.join(line.split())
        for j in self.js:
            ln = ln.replace(j, '')
        ln = ' '.join(ln.split())
        return ln

    def dejunk_text(self, txt):
        t = '\n'.join(splicer(txt))
        for p in eml_junk:
            t = p.sub('', t)
        t = defragment(t)
        t = '\n'.join(simple_replacer(t))
        for j in self.sorted_rejs:
            t = j.sub('', t)
        t = patch(t)
        ls = t.strip().splitlines()
        return '\n'.join(' '.join(l.split()) for l in ls).strip()


if __name__ == '__main__':
    j = Junk()
    j.junks_from('qnarre/junk.txt')
