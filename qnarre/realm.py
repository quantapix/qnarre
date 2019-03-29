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

import contextlib as cl
import collections.abc as abc

from itertools import chain

from .exporter import Exporter
from .nominals import para_join
from .base import config, num_to_name, lister
from .meta import converter, with_class_init, with_property
from .part import Defaults, Textual, Titled, Part, Contact, Role, Topic

realms = config.OPEN, config.PUBL, config.PROT, config.PRIV, config.ROOT


@with_class_init()
class Realm(Part):

    fixers = ()

    @classmethod
    def init(cls):
        cls.realms = {}
        for l, r in enumerate(realms):
            cls.realms[r] = cls(r, level=l)

    def __init__(self, label, level, **kw):
        super().__init__(label, **kw)
        self._level = level

    @property
    def level(self):
        return self._level


@cl.contextmanager
def realm_as(current):
    Realm.current = current
    yield
    del Realm.current


class Realmed(abc.MutableSequence):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._elems = [None] * len(Realm.realms)

    def __len__(self):
        return len(self._elems)

    def __iter__(self):
        return iter(self._elems)

    def __getitem__(self, i):
        return self._elems[i]

    def __setitem__(self, i, e):
        self._elems[i] = e

    def __delitem__(self, i):
        raise NotImplementedError

    def insert(self, i, e):
        raise NotImplementedError

    @property
    def realm(self):
        try:
            return Realm.current
        except AttributeError:
            return

    @property
    def current(self):
        c = None
        if self.realm:
            for i in range(self.realm.level + 1):
                e = self._elems[i]
                c = c if e is None else e
        return c

    @current.setter
    def current(self, e):
        self[self.realm.level] = e


class with_realm:

    default = None

    def __init__(self, name, creator, default=None):
        self.name = name
        self.multi = name.endswith('s')
        if default is not None:
            self.default = default
        elif self.multi:
            self.default = ()
        self.creator = creator

    def __call__(self, cls):
        d = self.default

        def getter(self):
            return self.current or d

        c = self.creator

        if self.multi:

            def setter(self, vs):
                self[self.realm.level] = None if vs is None else tuple(c(vs))
        else:

            def setter(self, v):
                self[self.realm.level] = None if v is None else c(v)

        setattr(cls, self.name, property(getter, setter))
        return cls


class Patched(Realmed):

    fixers = Realm.fixers
    patchers = ()

    @property
    def current(self):
        r = None
        if self.realm:
            for i in range(self.realm.level + 1):
                e = self._elems[i]
                if e is None:
                    if r is not None:
                        try:
                            r = self.patchers[i].patch(r)
                        except IndexError:
                            pass
                else:
                    r = e
        if r is not None:
            try:
                r = self.fixers[i].fix(r)
            except IndexError:
                pass
        return r


def parse_range(rng):
    def parse_one(r):
        ps = r.split('-')
        assert 0 < len(ps) < 3
        ps = [int(i) for i in ps]
        s = ps[0]
        e = s if len(ps) == 1 else ps[1]
        if s > e:
            e, s = s, e
        return range(s, e + 1)

    return set(chain(*[parse_one(r) for r in rng.split(',')]))


def get_list(realm, source, rng, ctxt, **_):
    if source:
        b = ctxt.base / ctxt.imgs_src
        d = (b / realm / source).with_suffix('')
        if d.exists():
            return sorted(str(p.relative_to(b)) for p in lister(d, rng))
    return ()


@with_realm('sources', Textual.creator)
class Sourced(Patched):
    def __init__(self, sources=(), **kw):
        super().__init__(**kw)
        self.sources = sources

    def convert_from(self, rec, **kw):
        s = rec.hdr.source or rec.source
        ps = s.split('[')
        if len(ps) == 2:
            s, r = ps
            r = set(num_to_name(i) for i in parse_range(r[:-1]))
        else:
            r = ()
        self.sources = get_list(str(self.realm), s, r, **kw)
        super().convert_from(rec, **kw)


@with_property('topic', Topic.create)
@with_realm('text', Textual.create)
class Subject(Part, Patched, Defaults):
    def __init__(self, label, topic=None, text=None, **kw):
        super().__init__(label, **kw)
        self.topic = topic
        self.text = text

    def __str__(self):
        return str(self.text or self.name)

    @property
    def group(self):
        return self.topic


@with_property('subject', Subject.create)
class About:
    def __init__(self, subject=None, **kw):
        super().__init__(**kw)
        self.subject = subject

    @property
    def subgroup(self):
        return self.subject

    def convert_from(self, rec, ctxt, **kw):
        n = rec.subject(ctxt)
        if n:
            s = self.subject
            if s:
                s.text = n
            else:
                t = rec.topic(ctxt)
                if t:
                    s = '{}_{}'.format(t, n)
                t = Topic.create(t, **kw)
                self.subject = Subject.create(s or n, topic=t, text=n, **kw)
        super().convert_from(rec, **kw, ctxt=ctxt)


@with_realm('texts', Textual.creator)
class Body(Patched):
    @classmethod
    def create(cls, i, **kw):
        return i if isinstance(i, cls) else cls(**kw)

    def __init__(self, texts=(), **kw):
        super().__init__(**kw)
        self.texts = tuple(texts)

    def __str__(self):
        return para_join(t.text for t in self.texts)

    @property
    def html(self):
        return Exporter.markdown.reset().convert(str(self))


@converter('Contact')
@with_realm('contact', Contact.create)
@with_property('body', Body.create)
class Agent(Part, Realmed, Titled, Defaults):

    role = None
    background = None
    justify = 0

    @classmethod
    def convert(cls, contact, ctxt=None, **kw):
        c = Contact.create(contact, **kw)
        r, b, j, s = config.all_traits.get(c.slug, (None, None, 0, c.slug))
        a = cls.create('a' + (s or c.slug), **kw)
        if not a[a.realm.level]:
            a.contact = c
            if r:
                a.role = Role.create(r, **kw)
            if b:
                a.background = b
            if j:
                a.justify = j
        return a

    def __init__(self, label, contact=None, body=None, **kw):
        super().__init__(label, **kw)
        self.contact = contact
        from django.utils.lorem_ipsum import sentence, paragraphs
        self.body = body or paragraphs(3)
        if not self.title:
            self.title = self.name.title()
        if not self.summary:
            self.summary = sentence().capitalize()

    @property
    def name(self):
        c = self.contact
        return c.name if c else super().name

    @property
    def subgroup(self):
        return self.role


@with_property('froms', Agent.creator)
@with_property('tos', Agent.creator)
@with_property('ccs', Agent.creator)
@with_property('bccs', Agent.creator)
class Sent:
    def __init__(self, froms=(), tos=(), ccs=(), bccs=(), **kw):
        super().__init__(**kw)
        self.froms = froms
        self.tos = tos
        self.ccs = ccs
        self.bccs = bccs

    def convert_from(self, rec, **kw):
        self.froms = (Agent.convert(c, **kw) for c in (rec.hdr.from_ or ()))
        self.tos = (Agent.convert(c, **kw) for c in (rec.hdr.to or ()))
        self.ccs = (Agent.convert(c, **kw) for c in (rec.hdr.cc or ()))
        self.bccs = (Agent.convert(c, **kw) for c in (rec.hdr.bcc or ()))
        super().convert_from(rec, **kw)
