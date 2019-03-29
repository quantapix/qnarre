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

from .blog import Blog
from .base import config
from .mboxes import Mboxes
from .meta import converter
from .context import Context
from .counter import counters
from .analyzer import Analyzer
from .log import Logger, start_stop_log
from .resource import Resource, resource
from .realm import Realm, realm_as, Agent
from .edit import protect, redact, obfuscate

log = Logger(__name__)


class Dispatch(Resource):

    _res_path = config.qnar_dst + 'dispatch.qnr'

    _blog = 'blog'
    _ctxt = None

    @classmethod
    def globals(cls):
        return globals()

    @property
    def ctxt(self):
        if self._ctxt is None:
            self._ctxt = Context.create(self.base, self.realm)
        return self._ctxt

    def filt_mbox(self, pool=None, **kw):
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Filtering '):
                if pool:
                    Mboxes(self.base).pool_filt(**kw)
                else:
                    Mboxes(self.base).filt_mbox(**kw)

    def merge_mbox(self, pool=None, wdir=None, **kw):
        wdir = wdir or config.ARCH
        wdir = config.recs_src + '/' + wdir + '/' + config.MBOX
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt, wdir=wdir)
            with start_stop_log(log, 'Merging '):
                if pool:
                    Mboxes(self.base).pool_merge(**kw)
                else:
                    Mboxes(self.base).merge_mbox(**kw)

    def strip_mbox(self, pool=None, wdir=None, **kw):
        wdir = wdir or config.ARCH
        wdir = config.recs_src + '/' + wdir + '/' + config.MBOX
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt, wdir=wdir)
            with start_stop_log(log, 'Stripping '):
                if pool:
                    Mboxes(self.base).pool_strip(**kw)
                else:
                    Mboxes(self.base).strip_mbox(**kw)

    def import_from(self, src, **kw):
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Importing from ' + src):
                ctxt.recs.import_from(src, **kw)

    def protect(self, **kw):
        with resource(self.ctxt) as ctxt:
            for n, t in config.bridge_aliases[self.realm]:
                ctxt.add_alias(n, t)
            r = config.PRIV
            with start_stop_log(log, 'Protecting from ' + r):
                s = Context.create(self.base, r)
                ctxt.recs.copy_from(s, protect, **kw, ctxt=ctxt)

    def redact(self, **kw):
        with resource(self.ctxt) as ctxt:
            for n, t in config.bridge_aliases[self.realm]:
                ctxt.add_alias(n, t)
            for r in (config.PROT, config.PRIV):
                with start_stop_log(log, 'Redacting from ' + r):
                    s = Context.create(self.base, r)
                    ctxt.recs.copy_from(s, redact, **kw, ctxt=ctxt)

    def obfuscate(self, **kw):
        with resource(self.ctxt) as ctxt:
            for n, t in config.bridge_aliases[self.realm]:
                ctxt.add_alias(n, t)
            for r in (config.PUBL, config.PROT, config.PRIV):
                with start_stop_log(log, 'Obfuscating from ' + r):
                    s = Context.create(self.base, r)
                    ctxt.recs.copy_from(s, obfuscate, **kw, ctxt=ctxt)

    def check_recs(self, **kw):
        a = Analyzer()
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Checking '):
                ms = ctxt.recs
                a.check_sanity(ms.grapher(**kw, links=None), **kw)
                a.check_coherence(ms.grapher(**kw, links=None), **kw)

    def graph_recs(self, **kw):
        pass
        # with resource(self.ctxt) as ctxt:
        # q = config.qnar_dst
        # with graph(ctxt.base / (q + '/qnarre.dot'), **kw) as g:
        # for c in TxtChains.creator(ctxt.recs):
        #    for n in c:
        #        print(n)

    def export_all(self, kind, **kw):
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Exporting ' + kind):
                src = self.base / (config.SRC + self.realm)
                dst = self.base / (config.DST + self.realm)
                if kind is config.ORGS:
                    from .images import Orgs
                    Orgs(src, dst).export_all(**kw)
                elif kind is config.IMGS:
                    from .images import Pngs
                    Pngs(src, dst).export_all(**kw)
                elif kind is config.PICS:
                    from .images import Jpgs
                    Jpgs(src, dst).export_all(**kw)
                elif kind is config.MBOX:
                    Mboxes(self.base).export_to(dst, **kw)
                elif kind is config.BLOG:
                    Blog(self.base).populate(dst, **kw)

    convert_args = ((('converted', '.'), ('failed', 'F')), 'Converting:')

    def convert(self, regy, **kw):
        with start_stop_log(log, 'Converting ' + self.realm.upper()):
            ctxt = self.ctxt
            with counters(self.convert_args, kw) as cs:
                with realm_as(Realm.realms[self.realm]):
                    for c in ctxt.contacts:
                        Agent.convert(c, regy=regy)
                        cs.incr('.')
                    for _, rs in ctxt.recs.chainer(**kw, ctxt=ctxt):
                        for r in rs:
                            converter.convert(r, regy=regy, ctxt=ctxt)
