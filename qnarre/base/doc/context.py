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

from .log import Logger
from .nominals import Nominals
from .resource import Resource
from .part import Contact, Place
from .base import rst_def, rst_ref, config

from .recs import Recs  # needed dynamically
from .part import Alias
from .resource import Mids
from .filters import Filters
from .content import Texts, Htmls, Attms
from .category import Subjects, Topics, Sources

log = Logger(__name__)

CTXT = config.CTXT


class Context(Resource):

    _assets = ('filters', 'recs', 'mids', 'topics', 'subjects', 'sources',
               'texts', 'htmls', 'attms')

    _res_path = config.qnar_dst + 'ctxt.qnr'

    _nominals = None
    _current = None
    _by_adr = None

    @classmethod
    def globals(cls):
        return globals()

    def __init__(self, elems=None, **kw):
        super().__init__(elems, **kw)
        if not elems and config.def_contacts[self.realm]:
            self.init = True
            self.slugs_for(config.def_contacts[self.realm])
            self.slugs_for(config.def_contacts[None])
            for n, t in (*config.contact_aliases[self.realm],
                         *config.contact_aliases[None]):
                self.add_alias(Contact.slugify(n), Contact.slugify(t))
            del self.init

    @property
    def assets(self):
        return (getattr(self, n) for n in self._assets)

    @property
    def loaded_assets(self):
        ns = self._assets
        return (a for a in (getattr(self, '_' + n, None) for n in ns) if a)

    @property
    def contacts(self):
        return (c for c in self.values() if isinstance(c, Contact))

    @property
    def places(self):
        return (p for p in self.values() if isinstance(p, Place))

    @property
    def nominals(self):
        if self._nominals is None:
            self._nominals = Nominals(''.join(e) for e in self.texts.elems)
        return self._nominals

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, current):
        if self._current:
            self.save()
        self._current = current

    @property
    def by_adr(self):
        if self._by_adr is None:
            self._by_adr = {}
            for c in self.contacts:
                c.map_by_adr(self)
        return self._by_adr

    def probe(self, adr):
        try:
            c = self.by_adr[adr]
        except KeyError:
            return None if hasattr(self, 'init') else self.filters.probe(adr)
        if config.EXCLUDED in self and c == self[config.EXCLUDED]:
            return False
        elif c != self[config.DEFAULT]:
            return True

    def slugs_for(self, spec, exclude=None, host=None):
        probe = None

        def by_names(names):
            ns = ','.join((names, host)) if host else names
            ss = [Contact.slugify(n) for n in ns.split(',') if n]
            try:
                ss = set(self[s].slug for s in ss)
            except KeyError:
                print(ss)
                raise
            if exclude:
                e = self[Contact.slugify(exclude)].slug
                if e in ss:
                    ss.remove(e)
            for s in ss:
                yield self[s]

        def by_hdr(hdr):
            ss = [(a.addr_spec.lower(), a.display_name) for a in hdr.addresses]
            ps = [self.probe(a) for a, _ in ss if a]
            nonlocal probe
            if any(ps):
                probe = True
            elif any([True for p in ps if p is False]):
                probe = False
            for a, n in ss:
                try:
                    c = self.by_adr[a]
                except KeyError:
                    s = Contact.slugify(n) if n else config.TBD
                    s = config.EXCLUDED if probe is False else s
                    i = self if hasattr(self, 'init') else None
                    try:
                        c = self[s]
                    except KeyError:
                        c = Contact(n, slug=s, adr=a, ctxt=i)
                    else:
                        c.append(a, i)
                yield c

        if spec is not None:
            if hasattr(spec, 'addresses'):
                cs = by_hdr(spec)
            else:
                cs = by_names(spec)
            ss = tuple(sorted(set(c.slug for c in cs)))
            return probe, ss
        return probe, ()

    def name(self, slug):
        try:
            n = self[slug].name
        except KeyError:
            n = slug
        # return '{} <{}@qnarre.com>'.format(n, slug)
        return str(n)

    def rename_msg(self, old, new):
        for a in self.loaded_assets:
            if hasattr(a, 'rename_msg'):
                a.rename_msg(old, new)

    def normalize_line(self, line):
        return line

    def extract(self, *args, text_only=False, **_):
        if not text_only:
            self.htmls.extract(*args)
            self.attms.extract(*args)
        return self.texts.extract(*args)

    def plainer(self, path, **kw):
        cs = sorted(self.contacts, key=lambda c: c.name)
        ps = sorted(self.places, key=lambda p: p.name)
        if path == CTXT:
            for e in (*cs, *ps):
                yield rst_def(CTXT, e.name)
                # yield from (' ' + l for l in e.plainer(**kw))
                yield from e.plainer(**kw)
        else:
            pre = CTXT + '/'
            assert path.startswith(pre)
            path = path[len(pre):]
            if path == 'people':
                yield from ('#. ' + rst_ref(CTXT, c.name) for c in cs)
            elif path == 'places':
                yield from ('#. ' + rst_ref(CTXT, p.name) for p in ps)
            elif path in self:
                yield rst_ref(CTXT, path)
            else:
                raise KeyError('{} not in ctxt'.format(path))

    def save(self, pref=None):
        pref = pref or self.current
        super().save(pref)
        for a in self.loaded_assets:
            a.save(pref)


for a in Context._assets:
    setattr(Context, '_' + a, None)

    def make_getter(name):
        n = '_' + name
        c = globals()[name.capitalize()]

        def get(self):
            if getattr(self, n) is None:
                setattr(self, n, c.create(self.base, self.realm))
            return getattr(self, n)

        return get

    setattr(Context, a, property(make_getter(a)))
