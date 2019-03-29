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


class Proof(Node):
    sign = '!p'
    authority = None

    def __init__(self,
                 text=None,
                 author=None,
                 agent=None,
                 authority=None,
                 factor=2,
                 **kw):
        super().__init__(factor=factor, **kw)
        if text:
            for k in ('factor', 'bias', 'weight'):
                kw.pop(k, None)
            self.claim = Claim(text=text, **kw)
            if not authority:
                if agent:
                    authority = 'agent'
                elif author:
                    authority = 'self'
        if authority:
            self.authority = Authority.create(name=authority)

    @property
    def weight(self):
        p = self.partial(self.authority.weight, self.claim.weight)
        return p + self.bias

    @property
    def credibility(self):
        return self.weight

    @property
    def value(self):
        a = self.authority.agency
        return '{} {}: {}'.format(super().value, a, self.claim.value)

    @property
    def fields(self):
        fs = self.authority.fields
        fs.update(self.claim.fields)
        fs.update(super().fields)
        fs['Credibility'] = self.credibility
        return fs
