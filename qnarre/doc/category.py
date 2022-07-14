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

from .junk import Junk
from .log import Logger
from .resource import Resource
from .base import config, LnkSubject, LnkTopic, LnkSource

from .part import slugify, Alias  # needed dynamically

log = Logger(__name__)


class Category(Resource):
    @classmethod
    def globals(cls):
        return globals()

    def __setitem__(self, k, name):
        ns = set(self.get(k, ()))
        ns.add(name)
        super().__setitem__(k, tuple(ns))

    def grapher(self, links=(), **kw):
        lk = self.link
        if links is not None and (not links or lk in links):
            for k, ns in self.items():
                if isinstance(ns, tuple):
                    for n in ns:
                        yield n, k, lk

    def rename_msg(self, old, new):
        def renamer():
            for k, v in self._elems.items():
                if isinstance(v, tuple):
                    yield k, tuple(n if n != old else new for n in v)
                else:
                    yield k, v

        self._elems = {k: v for k, v in renamer()}


class Subjects(Category):

    _res_path = config.qnar_dst + 'subjs.qnr'

    link = LnkSubject
    junk = Junk(('Re:', 'RE:', 'SAFE:', 'Fwd:', 'FW:', 'Fw:', '*'))

    @classmethod
    def dejunk(cls, txt, ctxt=None, **_):
        t = txt if txt else ''
        if t:
            t = cls.junk.dejunk_line(t)
            if ctxt:
                t = ctxt.normalize_line(t)
        return t

    def __init__(self, elems=None, **kw):
        super().__init__(elems, **kw)
        if not elems:
            for a in config.subject_aliases:
                self._elems.setdefault(slugify(a[1]), ())
                self.add_alias(*a)


class Topics(Category):

    _res_path = config.qnar_dst + 'topics.qnr'

    link = LnkTopic

    def __init__(self, elems=None, **kw):
        super().__init__(elems, **kw)
        if not elems:
            for a in config.topic_aliases:
                self._elems.setdefault(slugify(a[1]), ())
                self.add_alias(*a)


class Sources(Category):

    _res_path = config.qnar_dst + 'sources.qnr'

    link = LnkSource

    def res_ref(self, ref):
        print('res_ref', ref)
        ns = self.get(ref, ())
        if ns:
            return ns[0]
        log.warning('Failed to resolve source ref {}', ref)
        return ''
