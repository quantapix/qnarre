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

import pathlib as pth

from .. import create_from, load_from

from .author import Author
from .claim import Claim, Place
from .named import Named, Saved, Preset


class Boot(Preset):
    _nodes = None

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = ns = []
            for t, ps in self.props.items():
                for kw in ps:
                    ns.append(create_from(tag=t, **kw))
                    # print(ns[-1].name)
        return self._nodes

    def from_text(self, txt, **kw):
        super().from_text(txt, **kw)
        ts = list(self.props.pop('topic', ()))
        ns = list(self.props.pop('narrative', ()))
        for d in self.props.pop('narratives', ()):
            t, n = d.split(':')
            ts.append({'name': t})
            ns.append({'topic': t, 'name': n})
        self.props['topic'] = tuple(ts)
        self.props['narrative'] = tuple(ns)


class Net(Saved, Named):
    suff = '.py'
    docs = None
    org = None

    def __init__(self, *, root, docs=None, org=None, **kw):
        super().__init__(root=root, **kw)
        p = pth.Path('{}.boot'.format(self.name))
        self.boot = load_from(root=root, path=p)
        if docs:
            self.docs = docs
        if org:
            self.org = org

    def from_text(self, txt, preset=None, **kw):
        self.docs = preset or {}
        self.docs.update(eval(txt or '{}'))
        kw.update(tag='org', name=self.name, net=self)
        self.org = create_from(**kw)

    def to_text(self, **_):
        return repr(self.docs)

    def nodes(self):
        self.boot.nodes
        for d, gs in self.docs.items():
            d = create_from(tag='doc', name=d)
            an = d.author
            at = Author.create(name=an).tag
            for gi, rs in enumerate(gs, start=1):
                for ri, ns in enumerate(rs, start=1):
                    p = Place(d, gi, ri)
                    for kw in ns:
                        kw = kw.copy()
                        kw.setdefault(at, an)
                        kw.setdefault('tag', Claim.to_tag())
                        yield create_from(place=p, **kw)
