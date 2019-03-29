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
from .proof import Proof
from .narrative import Node


class Conjecture(Node):
    claims = proofs = None

    def __init__(self, text=None, proofs=None, **kw):
        super().__init__(**kw)
        if self.claims is None:
            self.claims, self.proofs = [], []
        if text:
            for k in ('factor', 'bias', 'weight'):
                kw.pop(k, None)
            self.claims.append(Claim(text=text, **kw))
        if proofs:
            ps = (p.strip() for p in proofs.split('|'))
            self.proofs.extend(Proof.create(name=p) for p in ps if p)

    @property
    def weight(self):
        cs = tuple(c.weight for c in self.claims)
        ps = tuple(p.weight for p in self.proofs)
        return self.partial(cs, ps) + self.bias


class Reality(Conjecture):
    sign = '!r'

    @property
    def coherence(self):
        return self.weight

    @property
    def value(self):
        return '{} C{}'.format(super().value, self.coherence)

    @property
    def fields(self):
        fs = super().fields
        fs['Reality'] = self.name
        ls = []
        for c in self.claims:
            fs2 = c.fields
            fs2.update(fs)
            fs2['Coherence'] = self.partial(c.weight)
            ls.append(fs2)
        for p in sorted(self.proofs, key=lambda p: p.sequence):
            fs2 = p.fields
            fs2['Topic'] = fs['Topic']
            fs2['Narrative'] = fs['Narrative']
            fs2['Reality'] = fs['Reality']
            fs2['Coherence'] = self.partial(p.weight)
            ls.append(fs2)
        return ls


class Dissent(Conjecture):
    agency = None

    @property
    def fragment(self):
        return self.weight

    @property
    def value(self):
        return '{} F{}'.format(super().value, self.fragment)

    @property
    def fields(self):
        fs = super().fields
        fs['Dissent'] = self.name
        ls = []
        for c in self.claims:
            fs2 = c.fields
            fs2.update(fs)
            fs2['Fragment'] = self.partial(c.weight)
            ls.append(fs2)
        for p in sorted(self.proofs, key=lambda p: p.sequence):
            fs2 = p.fields
            fs2['Topic'] = fs['Topic']
            fs2['Narrative'] = fs['Narrative']
            fs2['Dissent'] = fs['Dissent']
            fs2['Fragment'] = self.partial(p.weight)
            ls.append(fs2)
        return ls


class Misreading(Dissent):
    sign = '?m'
    _factor = 0


class Fairness(Dissent):
    sign = '?a'
    _factor = 0.25


class Negligence(Dissent):
    sign = '?n'
    _factor = 0.25


class Proportionality(Dissent):
    sign = '?y'
    _factor = 0.5


class Contradiction(Dissent):
    sign = '?x'
    _factor = 0.5


class Isolation(Dissent):
    sign = '?i'
    _factor = 0.75


class Omission(Dissent):
    sign = '?o'
    _factor = 0.75


class Distortion(Dissent):
    sign = '?s'
    _factor = 1
