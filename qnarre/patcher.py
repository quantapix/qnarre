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
import collections as co

from difflib import unified_diff

from .log import Logger
from .base import config
from .resource import Resource
from .nominals import flags, para_join, para_split

log = Logger(__name__)

fixes = ((r'(?P<lf>xxx?){2,}(?P<rt> ?(Date|Sent|To|Cc|Bcc|Subject): )',
          r'\g<lf>\g<rt>'), )


class Fixer:
    def __init__(self, fixes=(), **kw):
        super().__init__(**kw)
        self.fixes = fixes
        self.re_fixes = tuple((re.compile(flags + p), r) for p, r in fixes)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.fixes)

    def fix(self, txt):
        if isinstance(txt, tuple):
            return para_split(self.fix(para_join(txt)))
        for p, r in self.re_fixes:
            txt = p.sub(r, txt)
        return txt


class Fixers(Resource):

    _res_path = config.qnar_dst + 'fixers.qnr'

    @classmethod
    def globals(cls):
        return globals()


chunk_re = re.compile(r'^@@ -(\d+)(,(\d+))? \+(\d+)(,(\d+))? @@', re.ASCII)


class Chunk(co.namedtuple('Chunk', 'src tgt lns')):
    def applier(self, si, ti, src):
        ss = self.src.start
        while si < ss:
            i, sl = next(src)
            assert i == si
            yield sl
            si += 1
            ti += 1
        assert si == self.src.start and ti == self.tgt.start
        for l in self.lns:
            if l.startswith(('-', ' ')):
                i, sl = next(src)
                assert i == si and l[1:] == sl
                si += 1
                if l.startswith('-'):
                    continue
            else:
                assert l.startswith('+')
            yield l[1:]
            ti += 1
        assert si == self.src.stop and ti == self.tgt.stop
        return si


def diff_parser(udiff):
    c = ls = lt = None
    for ln in udiff:
        ln = ln or ' '
        m = chunk_re.match(ln)
        if m:
            if c:
                assert ls == c.src.stop and lt == c.tgt.stop
                yield c._replace(lns=tuple(c.lns))
            c = []
            for i in range(0, 4, 3):
                s = int(m.group(i + 1))
                n = int(m.group(i + 3)) if m.group(i + 3) else 1
                c.append(range(s, s + n))
            c = Chunk(*c, [])
            ls, lt = c.src.start, c.tgt.start
            continue
        elif c:
            if ln.startswith('-'):
                ls += 1
            elif ln.startswith('+'):
                lt += 1
            else:
                assert ln.startswith(' ')
                ls += 1
                lt += 1
            c.lns.append(ln)
    if c:
        assert ls == c.src.stop and lt == c.tgt.stop
        yield c._replace(lns=tuple(c.lns))


class Patcher:
    @classmethod
    def create(cls, src, dst):
        ud = unified_diff(src.splitlines(), dst.splitlines())
        cs = tuple(c for c in diff_parser(ud))
        return cls(cs)

    def __init__(self, chunks):
        super().__init__()
        self.chunks = chunks

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.chunks == other.chunks
        return NotImplemented

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.chunks)

    def patch(self, txt):
        if isinstance(txt, tuple):
            return para_split(self.patch(para_join(txt)))
        r = []
        si = ti = 1
        s = enumerate(txt.splitlines(), start=si)
        for c in self.chunks:
            a = c.applier(si, ti, s)
            while True:
                try:
                    r.append(next(a))
                except StopIteration as e:
                    si = e.value
                    break
            ti = len(r) + 1
        for _, l in s:
            r.append(l)
        return '\n'.join(r)


class Patchers(Resource):

    _res_path = config.qnar_dst + 'patchers.qnr'

    @classmethod
    def globals(cls):
        return globals()
