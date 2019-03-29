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

from .graph import feeder, sinker, bridger, Graphs
from .base import config, LnkQuoting, LnkReplying, Record
from .base import LnkTopic, LnkSubject, LnkProximity, LnkAudience, LnkSource


class Connects(Graphs):

    _graphs = tuple(
        a.label for a in (Record, LnkQuoting, LnkReplying, LnkTopic,
                          LnkSubject, LnkProximity, LnkAudience, LnkSource))

    def check(self):
        mn = set(self.record.nodes())
        if mn:
            for g in (self.quoting, self.replying):
                assert set(g.nodes()) <= mn
            for g in (self.topic, self.subject, self.proximity, self.audience,
                      self.source):
                for e in g.edges():
                    assert e[0] in mn

    def collapse_quoting(self, cntr, **_):
        qg, rg = self.quoting, self.replying
        dirty = None
        while True:
            for k in (feeder, sinker, bridger):
                for p, m, s in qg.linked_recs(k):
                    if p:
                        rg.add_edge(p, m)
                    if s:
                        rg.add_edge(m, s)
                    qg.remove_node(m)
                    cntr.incr('q')
                    dirty = True
            if not dirty:
                return dirty is False
            dirty = False

    def roll_up(self, msg, chn):
        mg = self.record
        c = mg.node[chn].setdefault(config.CHAIN, [])
        c.append(msg)
        c.extend(mg.node[msg].get(config.CHAIN, ()))
        for g in (self.proximity, self.audience):
            ss = [g.successors(m)[0] for m in (msg, chn) if m in g]
            if len(ss) != 2 or ss[0] != ss[1]:
                g.remove_msg(chn)
        for g in self.graphs:
            g.remove_msg(msg)

    def collapse(self, ms, cntr, **_):
        tg = self.topic
        ts = set(tg.successors(m)[0] for m in ms if m in tg)
        if len(ts) < 2:
            sg = self.subject
            ss = set(sg.successors(m)[0] for m in ms if m in sg)
            if len(ss) < 2:
                rg = self.replying
                p, m, s = ms
                if p and s:
                    c = p if rg.degree(p) < rg.degree(s) else s
                else:
                    c = p if p else s
                if ts:
                    tg.add_edge(c, ts.pop())
                if ss:
                    sg.add_edge(c, ss.pop())
                self.roll_up(m, c)
                cntr.incr('r')
                return True

    def collapse_replying(self, **kw):
        rg = self.replying
        dirty = None
        while True:
            for k in (feeder, sinker, bridger):
                for ms in rg.linked_recs(k):
                    if not self.collapse(ms, **kw):
                        rg.remove_node(ms[1])
                        continue
                    dirty = True
            if not dirty:
                self.check()
                return dirty is False
            dirty = False

    def collapse_loop(self, **kw):
        dirty = None
        while True:
            if self.collapse_quoting(**kw):
                dirty = True
            if self.collapse_replying(**kw):
                dirty = True
            if not dirty:
                self.check()
                return dirty is False
            dirty = False

    def collapse_proximity(self, cntr, **_):
        pg, rg = self.proximity, self.replying
        dirty = None
        for c in nx.weakly_connected_components(pg):
            p = None
            for m in sorted(m for m in c if not pg.in_degree(m)):
                if p:
                    rg.add_edge(p, m)
                    cntr.incr('p')
                    dirty = True
                p = m
        return dirty

    def collapse_audience(self, cntr, **_):
        ag, mg, pg = self.audience, self.record, self.proximity
        rg = self.replying
        dirty = None
        for c in nx.weakly_connected_components(ag):
            p = None
            for m in sorted(m for m in c if not ag.in_degree(m) and ':' in m):
                if m in pg and len(mg.node[m].get(config.CHAIN, ())) > 5:
                    p = None
                    continue
                if p:
                    rg.add_edge(p, m)
                    cntr.incr('a')
                    dirty = True
                p = m
        return dirty

    def collapse_subject(self, cntr, **_):
        sg, rg = self.subject, self.replying
        dirty = None
        for c in nx.weakly_connected_components(sg):
            p = None
            for m in sorted(m for m in c if not sg.in_degree(m) and ':' in m):
                if p:
                    rg.add_edge(p, m)
                    cntr.incr('s')
                    dirty = True
                p = m
        return dirty

    def collapse_all(self, **kw):
        self.purge_empty(**kw)
        self.collapse_loop(**kw)
        if self.collapse_proximity(**kw):
            self.collapse_loop(**kw)
        if self.collapse_audience(**kw):
            self.collapse_loop(**kw)
        if self.collapse_subject(**kw):
            self.collapse_loop(**kw)


Connects.init_class()
