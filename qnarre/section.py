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

from .realm import Agent
from .part import Setting
from .meta import with_property
from .message import StoryPost, BlogPost
from .message import Note, Message, Chain, Letter, Doc


class Section:

    incl = ()
    excl = ()
    hide = ()

    _extent = ()
    _groups = _subgroups = _parts = None

    def __init__(self, extent, **kw):
        super().__init__(**kw)
        self.extent = extent

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, ps):
        if ps:
            e = []
            for p in ps:
                c = type(p)
                if issubclass(c, self.incl) and not issubclass(c, self.excl):
                    if not p.parent:
                        e.append(p)
            self._extent = sorted(e)
        else:
            self.__dict__.pop('_extent', None)
        self.__dict__.pop('_groups', None)
        self.__dict__.pop('_subgroups', None)
        self.__dict__.pop('_parts', None)

    @property
    def groups(self):
        if self._groups is None:
            self.setup()
        return self._groups

    @property
    def subgroups(self):
        if self._subgroups is None:
            self.setup()
        return self._subgroups

    @property
    def parts(self):
        if self._parts is None:
            self.setup()
        return self._parts

    def setup(self):
        gs = []
        sg_g = {}
        ps_s = {}
        ps = set()
        for p in self.extent:
            s = p.subgroup
            if s:
                g = s.group
                if g and g.slug in self.hide:
                    p.hide = True
                    continue
                if g and g not in ps:
                    gs.append(g)
                    ps.add(g)
                if s not in ps:
                    sg_g.setdefault(g, []).append(s)
                    ps.add(s)
            ps_s.setdefault(s, []).append(p)
        self._groups = sorted(gs)
        for g, ss in sg_g.items():
            sg_g[g] = sorted(ss)
        self._subgroups = sg_g if self._groups else sg_g.get(None, ())
        # for s, ps in ps_s.items():
        #    ps_s[s] = sorted(ps)
        self._parts = ps_s if self._subgroups else ps_s.get(None, ())


class Story(Section):

    incl = (StoryPost, )
    hide = ('about', 'blurbs')


class Blog(Section):

    incl = (BlogPost, )


class Agents(Section):

    incl = (Agent, )


class Docs(Section):

    incl = (Note, Message, Chain, Letter, Doc)
    excl = (StoryPost, BlogPost)

    def update(self, settings):
        pass


@with_property('settings', Setting.creator)
class Session:
    def __init__(self, app, settings=(), **kw):
        super().__init__(**kw)
        self.parts_all = app.parts_all
        self.story = app.story
        self.blog = app.blog
        self.agents = app.agents
        self.docs = Docs(app.parts_flat)
        self.settings = settings

    def update(self):
        self.docs.update(self.settings)
