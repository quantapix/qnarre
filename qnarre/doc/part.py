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

import re

import collections.abc as abc

from unicodedata import normalize

from .base import config
from .meta import with_current, with_property


def slugify(value):
    v = str(value)
    v = normalize("NFKD", v).encode("ascii", "ignore").decode("ascii")
    v = re.sub(r"[^\w\s-]", "", v).strip().lower()
    return re.sub(r"[-\s]+", "-", v)


class Registry(abc.MutableMapping):
    def __init__(self, elems=None, **kw):
        super().__init__(**kw)
        self._elems = elems or {}

    def __bool__(self):
        return True

    def __len__(self):
        return len(self._elems)

    def __iter__(self):
        return iter(self._elems)

    def __getitem__(self, n):
        return self._elems[self.resolve_alias(n)]

    def __setitem__(self, n, e):
        self._elems[self.resolve_alias(n)] = e

    def __delitem__(self, n):
        del self._elems[self.resolve_alias(n)]

    def add_once(self, v):
        if v:
            try:
                v = self[v]
            except KeyError:
                self[v] = v
        return v

    def add_alias(self, name, tgt):
        return Alias(name, tgt, self)

    def resolve_alias(self, name):
        v = self._elems.get(slugify(name))
        while isinstance(v, Alias):
            name = v.tgt
            v = self._elems.get(name)
        return name

    def resolve_all(self, *names):
        for n in names:
            r = self.resolve_alias(n)
            if r != n:
                return r
        else:
            return n


@with_current()
class Labels(Registry):
    pass


@with_current()
class Texts(Registry):
    pass


class Parts(Registry):
    pass


class Defaults:

    parent = None

    tags = ()
    sources = ()

    def __init__(self, **kw):
        if kw:
            print(type(self))
            print(kw)
        super().__init__(**kw)

    def convert_from(self, src, **kw):
        pass


class Base:
    @classmethod
    def create(cls, i, **kw):
        return i if isinstance(i, cls) else cls(i, **kw)

    @classmethod
    def creator(cls, ii, **kw):
        for i in ii:
            yield cls.create(i, **kw)

    def __init__(self, **kw):
        super().__init__(**kw)

    @property
    def html(self):
        return str(self)


@with_property("text", Texts.current.add_once, "")
class Textual(Base):
    def __init__(self, text="", **kw):
        super().__init__(**kw)
        self.text = text

    def __str__(self):
        return self.text


class Title(Textual):
    pass


class Summary(Textual):
    pass


@with_property("title", Title.create)
@with_property("summary", Summary.create)
class Titled:
    def __init__(self, title=None, summary=None, **kw):
        super().__init__(**kw)
        self.title = title
        self.summary = summary

    def convert_from(self, rec, **kw):
        t = rec.hdr.title or ""
        if self.title:
            assert str(self.title) == t
        else:
            self.title = t
        s = rec.hdr.summary or ""
        if self.summary:
            assert str(self.summary) == s
        else:
            self.summary = s
        super().convert_from(rec, **kw)


@with_property("label", Labels.current.add_once, "")
class Part(Base):

    hide = False

    @classmethod
    def slugify(cls, label="", slug=""):
        return slug or slugify(label)

    @classmethod
    def create(cls, i, slug="", regy=None, **kw):
        if not isinstance(i, cls):
            s = cls.slugify(i, slug)
            i = i if regy is None else regy.get(s, i)
            if not isinstance(i, cls):
                i = cls(i, **kw, slug=s, regy=regy)
        return i

    @classmethod
    def get_template(cls):
        return "part"

    def __init__(self, label="", slug="", regy=None, default=False, **kw):
        super().__init__(**kw)
        self._slug = s = Labels.current.add_once(self.slugify(label, slug))
        if label != s:
            self.label = label
        if regy:
            assert s not in regy, "{} duplicate".format(s)
            if default and len(regy) == 0:
                regy[config.DEFAULT] = self
            if isinstance(self, Alias):
                regy._elems[s] = self
            else:
                regy[s] = self

    def __hash__(self):
        return hash(self._slug)

    def __eq__(self, other):
        if isinstance(other, Part):
            return self._slug == other._slug
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Part):
            return self._slug < other._slug
        return NotImplemented

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.name)

    def __str__(self):
        return self.name

    @property
    def slug(self):
        return self._slug

    @property
    def name(self):
        return self.label or self._slug

    def get_absolute_url(self):
        from django.urls import reverse

        return reverse("qnarre:part", kw={"slug": self._slug})


class Alias(Part):
    def __init__(self, name, tgt, regy=None):
        super().__init__(slugify(name), regy=regy)
        self.tgt = t = slugify(tgt)
        if regy:
            assert t in regy, "{} not found".format(t)

    def __repr__(self):
        return "{}, {!r})".format(super().__repr__()[:-1], self.tgt)


class Dated(Part):
    @property
    def date(self):
        s = self.slug.split("_")
        if len(s) == 2:
            if "-" in s[1]:
                return s[0] + " at " + s[1].replace("-", ":")
        return s[0]


class Tag(Part):
    pass


@with_property("tags", Tag.creator)
class Tagged:
    def __init__(self, tags=(), **kw):
        super().__init__(**kw)
        self.tags = tags


class Role(Part):

    group = None


class Topic(Part):
    @property
    def name(self):
        return (super().name or config.TBD).title()


class Contact(Part):

    adrs = ()
    _adr = None

    @classmethod
    def slugify(cls, name="", slug=""):
        if not slug:
            n = name.strip() if name else ""
            s = n.split()
            if len(s) == 2:
                f, la = s
            elif len(s) >= 3:
                f, _, la = s[:3]
            else:
                return slugify(n)
            assert f and la
            if f in ("Atty.", "Dr.", "Ms.", "Mrs.", "Mr.", "Sr.", "Hon."):
                return slugify("{}_{}".format(f[:-1], la))
            return slugify("{}_{}".format(f[0], la))
        return slug

    def __init__(self, label, adr=None, ctxt=None, regy=None, **kw):
        super().__init__(label, **kw, regy=regy or ctxt, default=True)
        if ctxt:
            self.append(adr, ctxt)
        else:
            self._adr = adr or ()

    def __repr__(self):
        a = self._adr
        return "{}, {!r})".format(super().__repr__()[:-1], self.adrs if a is None else a)

    def map_by_adr(self, ctxt):
        if self._adr is not None:
            a = self._adr
            del self._adr
            self.append(a, ctxt)

    def append(self, adr, ctxt):
        if isinstance(adr, tuple):
            for a in adr:
                self.append(a, ctxt)
        elif adr and ctxt:
            adr = Labels.current.add_once(adr.lower())
            try:
                assert self == ctxt.by_adr[adr]
            except KeyError:
                ctxt.by_adr[adr] = self
            self.adrs = (*self.adrs, adr)

    def plainer(self, **kw):
        yield ":Name: {}".format(self.name)
        yield ":Email: {}".format("xyz@aaa.com")
        yield ":Web site: {}".format("www.abc.com")
        yield ":Archived page: {}".format("page...")


@with_property("address", Labels.current.add_once, "")
class Place(Part):
    def __init__(self, name, address=None, ctxt=None):
        super().__init__(name, regy=ctxt)
        self.address = address

    def __repr__(self):
        return "{}, {!r})".format(super().__repr__()[:-1], self.address)


@with_property("value", Textual.create)
class Setting(Part):
    def __init__(self, name, value="", **kw):
        super().__init__(name, **kw)
        self.value = value

    @property
    def html(self):
        return self.name + ": " + self.value
