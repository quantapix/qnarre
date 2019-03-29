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

from .claim import Claim
from .narrative import Node
from .author import Authority
from .conflict import Conflict
from .conjecture import Dissent


class Judgment(Node):
    claims = conflicts = dissents = None

    def __init__(self,
                 text=None,
                 conflicts=None,
                 dissents=None,
                 authority=None,
                 **kw):
        super().__init__(**kw)
        if self.claims is None:
            self.claims, self.conflicts, self.dissents = [], [], []
        if text:
            for k in ('factor', 'bias', 'weight'):
                kw.pop(k, None)
            self.claims.append(Claim(text=text, **kw))
        if conflicts:
            fs = (f.strip() for f in conflicts.split('|') if ':' in f)
            self.conflicts.extend(Conflict.create(name=f) for f in fs if f)
        if dissents:
            ds = (d.strip() for d in dissents.split('|') if ':' in d)
            self.dissents.extend(Dissent.create(name=d) for d in ds if d)
        if authority:
            self.authority = Authority.create(name=authority)

    @property
    def weight(self):
        cs = tuple(c.weight for c in self.claims)
        fs = tuple(f.weight for f in self.conflicts)
        ds = tuple(d.weight for d in self.dissents)
        return self.partial(cs, fs, ds) + self.bias

    @property
    def turmoil(self):
        return self.weight

    @property
    def value(self):
        t = self.turmoil
        return '{} {}: T{}'.format(super().value, self.authority.agency, t)

    @property
    def fields(self):
        fs = super().fields
        fs['Judgment'] = self.name
        ls = []
        for c in self.claims:
            fs2 = c.fields
            fs2.update(fs)
            fs2['Turmoil'] = self.partial(c.weight)
            ls.append(fs2)
        for f in sorted(self.conflicts, key=lambda f: f.sequence):
            fs2 = f.fields
            fs2['Topic'] = fs['Topic']
            fs2['Narrative'] = fs['Narrative']
            fs2['Judgment'] = fs['Judgment']
            fs2['Turmoil'] = self.partial(f.weight)
            ls.append(fs2)
        for d in sorted(self.dissents, key=lambda d: d.sequence):
            fs2 = d.fields
            fs2['Topic'] = fs['Topic']
            fs2['Narrative'] = fs['Narrative']
            fs2['Judgment'] = fs['Judgment']
            fs2['Turmoil'] = self.partial(d.weight)
            ls.append(fs2)
        return ls


class Validation(Judgment):
    sign = '=v'
    _factor = 0


class Confusion(Judgment):
    sign = '=c'
    _factor = 0.25


class Bias(Judgment):
    sign = '=b'
    _factor = 0.5


class Disregard(Judgment):
    sign = '=g'
    _factor = 0.75


class Fabrication(Judgment):
    sign = '=f'
    _factor = 1
