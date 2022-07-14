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

from ..edit import fudge

from .named import Tagged


class Claim(Tagged):
    sign = '~'
    loss = 0

    def __init__(self, *, text, place, loss=None, **_):
        super().__init__(tag=self.tag)
        self.text = fudge() if text == 'fudge' else text
        self.place = place
        if loss is not None:
            self.loss = loss

    def __str__(self):
        return '({}) {}'.format(self.sign, self.value)

    @property
    def factor(self):
        return self.place.doc.factor * super().factor

    @property
    def bias(self):
        return self.place.doc.bias + super().bias

    @property
    def credibility(self):
        return self.weight

    @property
    def value(self):
        c = '#{}'.format(self.credibility) if self.credibility != 1 else ''
        a = ' ${}'.format(self.loss) if self.loss else ''
        return '{}"{}"{} {}'.format(c, self.text, a, self.place)

    @property
    def fields(self):
        fs = {
            'Text': self.text,
            'Credibility': self.credibility,
            'Loss': self.loss
        }
        fs.update(self.place.fields)
        fs['Type'] = self.tag
        return fs


class Place(Tagged):
    def __init__(self, doc, page, para):
        super().__init__(tag=self.tag)
        self.doc = doc
        self.page = page
        self.para = para

    def __str__(self):
        return '@ {}:{}:{}'.format(self.doc.name, self.page, self.para)

    @property
    def fields(self):
        fs = {'Page': self.page, 'Para': self.para}
        fs.update(self.doc.fields)
        fs['Type'] = self.tag
        return fs
