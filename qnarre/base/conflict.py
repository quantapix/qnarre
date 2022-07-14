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
from .author import Agent
from .narrative import Node
from .conjecture import Reality


class Conflict(Node):
    agency = None
    count = 1

    def __init__(self, text=None, reality=None, agency=None, **kw):
        super().__init__(**kw)
        if text:
            for k in ('factor', 'bias', 'weight'):
                kw.pop(k, None)
            self.claim = Claim(text=text, **kw)
        if reality:
            self.reality = Reality.create(name=reality)
        if agency:
            self.agency = Agent.create(name=agency)

    @property
    def sign(self):
        return ('?' if self.agency else '') + self._sign

    @property
    def topic(self):
        return super().topic or self.reality.topic

    @property
    def narrative(self):
        return super().narrative or self.reality.narrative

    @property
    def factor(self):
        return super().factor * self.count * (2 if self.agency else 1)

    @property
    def weight(self):
        return self.partial(self.reality.weight) + self.bias

    @property
    def fragment(self):
        return self.weight

    @property
    def value(self):
        v = super().value
        n = self.reality.name
        f = self.fragment
        return '{} {} <-> {} F{}'.format(v, self.claim.value, n, f)

    @property
    def fields(self):
        fs = {'Reality': self.reality.name, 'Fragment': self.fragment}
        fs.update(self.claim.fields)
        fs.update(super().fields)
        if self.agency:
            fs['Agency'] = self.agency.value
        return fs


class Inherent(Conflict):
    _factor = 0
    _sign = '?h'


class Conceal(Conflict):
    _factor = 0.25
    _sign = '?c'


class Deceive(Conflict):
    _factor = 0.5
    _sign = '?d'


class Fraud(Conflict):
    _factor = 0.75
    _sign = '?u'


class Extort(Conflict):
    _factor = 1
    _sign = '?e'


class Repeat(Claim):
    sign = '*n'

    def __init__(self, conflict, **kw):
        super().__init__(**kw)
        assert ':' in conflict
        self.conflict = Conflict.create(name=conflict)
        self.conflict.count += 1

    @property
    def value(self):
        return '{} +{}'.format(super().value, self.conflict.name)

    @property
    def fields(self):
        fs = self.conflict.fields
        fs.update(super().fields)
        fs['Name'] = self.conflict.name
        fs['Type'] = self.tag
        return fs
