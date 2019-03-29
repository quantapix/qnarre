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
from .judgment import Judgment


class Activism(Node):
    claims = judgments = None

    def __init__(self, text=None, judgments=None, authority=None, **kw):
        super().__init__(**kw)
        if self.claims is None:
            self.claims, self.judgments = [], []
        if text:
            for k in ('factor', 'bias', 'weight'):
                kw.pop(k, None)
            self.claims.append(Claim(text=text, **kw))
        if judgments:
            js = (j.strip() for j in judgments.split('|') if ':' in j)
            self.judgments.extend(Judgment.create(name=j) for j in js if j)
        if authority:
            self.authority = Authority.create(name=authority)

    @property
    def weight(self):
        cs = tuple(c.weight for c in self.claims)
        js = tuple(j.weight for j in self.judgments)
        return self.partial(cs, js) + self.bias

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
        fs['Activism'] = self.name
        ls = []
        for c in self.claims:
            fs2 = c.fields
            fs2.update(fs)
            fs2['Turmoil'] = self.partial(c.weight)
            ls.append(fs2)
        for j in sorted(self.judgments, key=lambda j: j.sequence):
            fs2 = c.fields
            fs2['Topic'] = fs['Topic']
            fs2['Narrative'] = fs['Narrative']
            fs2['Activism'] = fs['Activism']
            fs2['Turmoil'] = self.partial(j.weight)
            ls.append(fs2)
        return ls


class Exclude(Activism):
    sign = '@x'


class Insinuate(Activism):
    sign = '@i'


class Polarize(Activism):
    sign = '@o'


class Recast(Activism):
    sign = '@r'


class Elevate(Activism):
    sign = '@e'


class Victimize(Activism):
    sign = '@v'


class Exploit(Activism):
    sign = '@t'


class Perpetuate(Activism):
    sign = '@p'
