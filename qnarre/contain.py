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

import networkx as nx

from .graph import Graphs
from .nominals import nominal
from .base import Record, LnkFull, LnkPartial


def is_partial(src, dst, sample=20, chunk=120):
    n = len(src)
    if n > chunk and len(dst) > chunk:
        if src[:sample] in dst or src[-(sample + 1):-1] in dst:
            return True
        ms, mc = sample // 2, chunk // 2
        for i in range(n // chunk):
            i *= chunk
            s = src[i:i + chunk]
            if len(s) == chunk:
                if s[mc - ms:mc + ms] in dst:
                    return True


class Contains(Graphs):

    _graphs = tuple(a.label for a in (Record, LnkFull, LnkPartial))

    def msg_attrs(self, txt, kind, **kw):
        n = nominal(txt)
        kw.update(empty=len(n) < 5, nominal=n, kind=kind)
        return kw

    def grow_full(self, cntr, **_):
        mg, fg = self.record, self.full
        ns = ((m, mg.node[m]['nominal']) for m in mg.nodes())
        ns = sorted(ns, key=lambda t: len(t[1]))
        for i, (m, n) in enumerate(ns):
            ns2 = ns[i + 1:]
            while ns2:
                m2, n2 = ns2.pop(0)
                if n in n2:
                    fg.add_edge(m, m2)
                    if len(n) == len(n2):
                        fg.add_edge(m2, m)
                        cntr.incr('=')
                    else:
                        cntr.incr('<')
                    ss = nx.dfs_successors(fg, m2).values()
                    ss = {s for sl in ss for s in sl}
                    ns2 = [(m, n) for m, n in ns2 if m not in ss]

    def grow_partial(self, cntr, **_):
        mg, fg, pg = self.record, self.full, self.partial
        ns = [(m, mg.node[m]['nominal']) for m in mg.nodes()]
        for i, (m, n) in enumerate(ns):
            ns2 = ns[:]
            del ns2[i]
            while ns2:
                m2, n2 = ns2.pop(0)
                if is_partial(n, n2):
                    pg.add_edge(m, m2)
                    cntr.incr('~')
                    ss = nx.dfs_successors(fg, m2).values()
                    ss = {s for sl in ss for s in sl}
                    ns2 = [(m, n) for m, n in ns2 if m not in ss]

    def grow_from(self, src, **kw):
        super().grow_from(src, **kw)
        self.purge_empty(**kw)
        self.grow_full(**kw)
        # self.grow_partial(**kw)


Contains.init_class()
