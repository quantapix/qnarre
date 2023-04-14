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

from .base import config


def para_make(txt):
    ps = (p.strip() for p in txt.strip().split('\n\n') if p)
    ps = (' '.join(p.split()).strip() for p in ps if p)
    return '\n\n'.join(p for p in ps if p)


def para_split(txt):
    return txt.split('\n\n') if txt else ()


def para_join(paras):
    return '\n\n'.join(paras)


flags = r'(?aim)'

chars = r'[^a-z0-9]'
chars = re.compile(flags + chars)

nbs = r'[^<>\n]'


def nominal(txt):
    t = txt.lower()
    for p in tuple(re.compile(flags + p) for p in config.nominal_offs):
        t = p.sub('', t)
    return ''.join(chars.split(t))


def compare(left, right):
    if left == right:
        return config.EQ
    left, right = nominal(left), nominal(right)
    if left == right:
        return config.EQ
    if right.startswith(left):
        return config.LT
    if left.startswith(right):
        return config.GT


fields = r'^ ?(date|subject|from|to|cc|bcc|in-reply-to|message-id): '
fields = re.compile(flags + fields)

om = r'^ ?-----Original_Message-----$'
om = re.compile(flags + om)

eom = r'^ ?-----End_Original_Message-----$'
eom = re.compile(flags + eom)

begin = r'^ ?(on .+?wrote:$)|from: '
begin = re.compile(flags + begin)


def quoter(lines, level=0):
    qs = []
    nqs = []
    quoting = None if level else False
    fwd = False
    for ln in lines:
        if ln.startswith('>'):
            if ln.startswith('>>'):
                qs.append(ln[1:])
            else:
                ln = ln[2:] if ln.startswith('> ') else ln[1:]
                if not quoting and not fwd:
                    if fields.match(ln) or om.match(ln) or begin.match(ln):
                        fwd = True
                    else:
                        nqs.append('| ' + ln)
                        continue
                qs.append(ln)
        elif quoting:
            if eom.match(ln):
                quoting = False
            elif not om.match(ln):
                qs.append(ln)
        elif om.match(ln):
            quoting = True
        elif begin.match(ln):
            if quoting is None:
                quoting = False
                nqs.append(ln)
            else:
                quoting = True
                qs.append(ln)
        else:
            nqs.append(ln)
    nqs = '\n'.join(nqs).strip().splitlines()
    if nqs:
        yield level, nqs
    qs = '\n'.join(qs).strip().splitlines()
    if qs:
        yield from quoter(qs, level=level + 1)


ow = r'(On .+?wrote:\n)|(Original_Message)'
ow = re.compile(flags + ow)


class Nominals:
    def __init__(self, txts):
        self.seq = ''.join(nominal(t) for t in txts)

    def __contains__(self, txt):
        return txt and (nominal(txt) in self.seq
                        or nominal(ow.split(txt, maxsplit=1)[0]) in self.seq)

    def append(self, txt):
        n = nominal(txt)
        if n and n not in self.seq:
            self.seq += n
            return n
