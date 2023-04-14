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

from .base import Record

feeder = 0, 1
bridger = 1, 1
sinker = 1, 0


class DiGraph(nx.DiGraph):
    def empty_recs(self):
        for m, d in dict(self.nodes(data=True)).items():
            if d.get('empty', False):
                yield m

    def linked_recs(self, kind):
        i, o = kind
        ms = (m for m, d in self.in_degree() if d == i)
        for m in (m for m, d in self.out_degree(ms) if d == o):
            if m in self:
                if self.in_degree(m) == i and self.out_degree(m) == o:
                    p = self.predecessors(m)[0] if i else None
                    s = self.successors(m)[0] if o else None
                    yield p, m, s

    def purge_recs(self):
        if self.size():
            for m in self.nodes():
                if not self.degree(m):
                    self.remove_node(m)

    def remove_msg(self, msg):
        if msg in self:
            for p in self.predecessors(msg):
                for s in self.successors(msg):
                    self.add_edge(p, s)
            self.remove_node(msg)


class Graphs:
    @classmethod
    def init_class(cls):
        for g in cls._graphs:
            setattr(cls, '_' + g, None)

            def make_getter(name):
                n = '_' + name

                def get(self):
                    if getattr(self, n) is None:
                        setattr(self, n, DiGraph())
                    return getattr(self, n)

                return get

            setattr(cls, g, property(make_getter(g)))

    def __init__(self, seed=(), **kw):
        super().__init__(**kw)
        for i in seed:
            self.add_item(i)

    @property
    def graphs(self):
        return (getattr(self, n) for n in self._graphs)

    def msg_attrs(self, txt, kind, **kw):
        kw.update(empty=not bool(txt), kind=kind)
        return kw

    def add_item(self, item, cntr, **kw):
        f, s, k = item
        if issubclass(k, Record):
            self.record.add_node(f, **self.msg_attrs(s, k, **kw))
            cntr.incr('record')
        else:
            getattr(self, k.label).add_edge(f, s)
            cntr.incr(k.label)

    def check(self):
        pass

    def grow_from(self, src, adjs=None, **kw):
        for i in src:
            self.add_item(i, **kw)
        if adjs:
            self.adjust_from(adjs)
        self.check()

    def purge_empty(self, cntr, **_):
        for m in self.record.empty_recs():
            for g in self.graphs:
                g.remove_msg(m)
                cntr.incr('d')
        self.check()


"""
    @property
    def comps(self):
        return nx.weakly_connected_components(self.nxdg)

    def roots(self, comp):
        return sorted(n for n, d in self.nxdg.in_degree(comp) if not d)

    def nodes(self, root):
        return nx.dfs_preorder_nodes(self.nxdg, root)

    def merge(self, other):
        return self

    def init_from(self, src):
        pass

    def adjust_from(self, src):
        pass


import contextlib as cl

@cl.contextmanager
def graph(path, directed=True, **kw):
    g = nx.DiGraph(**kw) if directed else nx.Graph(**kw)
    yield g
    a = nx.nx_agraph.to_agraph(g)
    #p = str(path)
    # g.write_dot(p)
    #g = gv.AGraph()
    # g.read(p)
    a.draw(str(path.with_suffix('.png')), prog="neato")


import pygraphviz as gv
 A = gv.AGraph()

A.node_attr['style'] = 'filled'
A.node_attr['shape'] = 'circle'
A.node_attr['fixedsize'] = 'true'
A.node_attr['fontcolor'] = '#FFFFFF'

for i in range(16):
    A.add_edge(0,  i)
    n = A.get_node(i)
    n.attr['fillcolor'] = "#%2x0000" % (i * 16)
    n.attr['height'] = "%s" % (i / 16.0 + 0.5)
    n.attr['width'] = "%s" % (i / 16.0 + 0.5)

print(A.string())
A.write("/tmp/star.dot")
print("Wrote star.dot")
A.draw('/tmp/star.png', prog="circo")
print("Wrote star.png")
"""
