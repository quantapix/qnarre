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

from .named import Named


class Author(Named):
    title = ''
    upper = None

    def __init__(self, *, title=None, upper=None, **kw):
        super().__init__(**kw)
        if title:
            self.title = title.title()
        if upper:
            self.upper = upper

    @property
    def value(self):
        n = self.name.replace('-', ' ')
        n = n.upper() if self.upper else n.title()
        n = n.replace(' ', '-')
        return self.title + ' ' + n

    @property
    def fields(self):
        fs = {'Author': self.value}
        fs.update(super().fields)
        return fs


class Agent(Author):
    def __init__(self, *, kind=None, **kw):
        super().__init__(**kw)
        if kind:
            self.kind = kind
            self.also_as(Author.to_tag())

    @property
    def fields(self):
        fs = {'Kind': self.kind}
        fs.update(super().fields)
        return fs


class Authority(Agent):
    agency = 'Sworn'

    def __init__(self, *, agency=None, **kw):
        super().__init__(**kw)
        if agency:
            self.agency = agency.title()

    @property
    def fields(self):
        fs = {'Agency': self.agency}
        fs.update(super().fields)
        return fs
