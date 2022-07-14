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

from .. import to_class
from ..rectify import QNERR


class Tagged:
    _factor = _weight = 1
    _bias = 0

    def __init__(self, *, tag, factor=None, bias=None, weight=None, **kw):
        assert tag == self.tag
        for k in ('agent', 'author', 'authority', 'genre', 'preset', 'place'):
            kw.pop(k, None)
        if kw:
            print(kw)
        super().__init__(**kw)
        if factor is not None:
            self._factor = factor
        if bias is not None:
            self._bias = bias
        if weight is not None:
            self._weight = weight

    @classmethod
    def to_tag(cls):
        return cls.__name__.lower()

    @property
    def tag(self):
        return type(self).to_tag()

    @property
    def factor(self):
        return self._factor

    @property
    def bias(self):
        return self._bias

    @property
    def weight(self):
        return self.factor * self._weight + self.bias

    def partial(self, *parts):
        ws = []
        for p in parts:
            if isinstance(p, tuple):
                ws.extend(p)
            else:
                ws.append(p)
        return self.factor * (sum(ws) if self._weight == 1 else len(ws))


class Saved:
    root = None

    def __init__(self, *, root=None, path=None, text=None, **kw):
        super().__init__(**kw)
        if root:
            self.root = root
        if text is None and path:
            text = ''
            p = self.root / path
            if p.exists():
                text = p.read_text(encoding='ascii', errors=QNERR)
        if text is not None:
            self.from_text(text, root=root, **kw)

    def __str__(self):
        return '{}{}'.format(self.name, self.suff)

    @property
    def suff(self):
        return '.' + self.tag

    @property
    def path(self):
        return self.root / str(self)

    def save(self, **kw):
        t = self.to_text(**kw)
        self.path.write_text(t, encoding='ascii', errors=QNERR)


class Named(Tagged):
    sequence = None

    _by_name = {}
    _by_tag = None

    _seq = 0

    @classmethod
    def next_seq(cls):
        cls._seq += 1
        return cls._seq

    @classmethod
    def next_name(cls):
        return '{:0>6d}'.format(len(cls._by_name))

    @classmethod
    def create(cls, *, name, tag=None, **kw):
        n = name if ':' in name else (':' + name)
        t, n = n.split(':')
        t = tag if tag else t
        t = t if t else cls.to_tag()
        n = cls.next_name() if n == 'fudge' else n
        k = t + ':' + n
        try:
            v = cls._by_name[k]
            if len(kw):
                v.__init__(tag=t, name=n, **kw)
        except KeyError:
            c = cls if t == cls.to_tag() else to_class(t)
            if not len(kw):
                kw['empty'] = True
            cls._by_name[k] = v = c(tag=t, name=n, **kw)
            cls._by_tag = None
        return v

    @classmethod
    def by_tag(cls, tag):
        if cls._by_tag is None:
            cls._by_tag = bt = {}
            for n in cls._by_name.values():
                bt.setdefault(n.tag, []).append(n)
        return cls._by_tag.get(tag, ())

    def __init__(self, *, name, empty=False, **kw):
        super().__init__(**kw)
        self.name = name
        if not empty and self.sequence is None:
            self.sequence = self.next_seq()

    def __str__(self):
        return "'{}:{}'".format(self.tag, self.name)

    @property
    def fields(self):
        return {'Type': self.tag, 'Name': self.name}

    def also_as(self, tag):
        k = tag + ':' + self.name
        assert k not in self._by_name
        self._by_name[k] = self


class Preset(Saved, Named):
    props = {}

    def from_text(self, txt, **_):
        self.props = eval(txt or '{}')
