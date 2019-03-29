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
import importlib as imp

from .. import load_from

from .doc import Doc
from .net import Net
from .stats import Stats
from .named import Named, Saved


class Org(Saved, Named):
    _tags = None
    docs = None
    net = None

    def __init__(self, *, root, docs=None, net=None, **kw):
        super().__init__(root=root, **kw)
        p = pth.Path('{}.preset'.format(self.name))
        self.preset = load_from(root=root, path=p)
        if docs:
            self.docs = docs
        if net:
            self.net = net

    @property
    def tags(self):
        if self._tags is None:
            m = imp.import_module('qnarre.core')
            ps = []
            for n in getattr(m, 'all_nodes'):
                if isinstance(n, tuple):
                    ps.append((n[0].to_tag(), n[1]))
                else:
                    n = n.to_tag()
                    ps.append((n, n[0]))
            self._tags = ' '.join('{}({})'.format(p[0], p[1]) for p in ps)
        return self._tags

    def from_text(self, txt, *, root, **_):
        ds, nds = [], {}
        for d in txt.split('\n* ')[1:]:
            d = d.split('\n', 1)[1]
            dps, d = d.split('\n    :END', 1)
            dps = _to_props(dps.split(':PROPERTIES:', 1)[1])
            dps['pages'] = gs = []
            ngs = []
            for g in d.split('\n** Page')[1:]:
                rs, nrs = [], []
                for r in g.split('\n*** Para')[1:]:
                    r = r.split(':TEXT:', 1)[1]
                    r, ns = r.split('\n    :END', 1)
                    rs.append(r.strip().splitlines())
                    nps = []
                    for n in ns.split('\n**** ')[1:]:
                        ps = n.split('\n', 1)
                        ts = ps.pop(0)
                        ps = ps[0].strip() if ps else ''
                        if ps:
                            ps = ps.split('\n    :END', 1)[0]
                            ps = _to_props(ps.split(':PROPERTIES:', 1)[1])
                        else:
                            ps = {}
                        ts = ts.split(':')
                        ps['text'] = ts.pop(0).strip()
                        if ts:
                            ps['tag'] = ' '.join(ts).strip()
                        nps.append(ps)
                    nrs.append(nps)
                gs.append(rs)
                ngs.append(nrs)
            ds.append(Doc.create(**dps, root=root))
            nds[dps['name']] = ngs
        self.docs = ds
        self.net = Net.create(name=self.name, docs=nds, root=root)

    def to_text(self, **_):
        txt = [
            '#+TITLE: {}\n'.format(self.name),
            '#+TAGS: {}'.format(self.tags),
            '#+COLUMNS: %ONE(No.) %TOPIC(Topic) %ITEM(Subject Summary)',
            '#+PROPERTY: ONE 1',
            '#+TODO: TODO | DONE',
            '',
        ]
        pres = self.preset.props
        ss = Stats(*self.net.docs.get('stats', ()))
        for d in self.docs:
            f = '* {} {}\n{}'
            txt.append(f.format(d.date, d.title, _to_text(d.props)))
            ngs = self.net.docs.get(d.name, ())
            for gi, rs in enumerate(d.pages):
                txt.append('** Page {}'.format(gi + 1))
                nrs = ngs[gi] if gi < len(ngs) else ()
                for ri, ls in enumerate(rs):
                    txt.append('*** Para {}\n    :TEXT:'.format(ri + 1))
                    txt.extend(ls)
                    txt.append('    :END:')
                    ns = nrs[ri] if ri < len(nrs) else _to_nodes(ls, ss, pres)
                    for ps in ns:
                        ps = ps.copy()
                        t = ps.pop('text')
                        ts = ps.pop('tag', '').strip().replace(' ', ':')
                        ts = ' :{}:'.format(ts) if ts else ''
                        ps = ('\n' + _to_text(ps)) if ps else ''
                        txt.append('**** {}{}{}'.format(t, ts, ps))
        self.net.docs['stats'] = ss.as_tuple
        print(ss)
        return '\n'.join(txt).strip()


def _to_text(props):
    ls = ['    :PROPERTIES:']
    m = 1 + max(len(k) for k in props.keys())
    for k, v in props.items():
        if v is not None:
            ls.append('    :{}:{}{}'.format(k, ' ' * (m - len(k)), v))
    ls.append('    :END:')
    return '\n'.join(ls)


def _to_props(txt):
    ps = {}
    for ln in txt.strip().splitlines():
        _, k, v = ln.split(':', 2)
        k, v = k.strip(), v.strip()
        if k and v:
            ps[k] = v
    return ps


def _to_nodes(lines, stats, preset):
    for ln in lines:
        ps = {'text': ln}
        if ln.startswith('@'):
            cs = ln.split(':', 2)
            if len(cs) > 1:
                c, arg1, ln = cs
                ln = ln.strip()
            else:
                c, ln = ln.split(None, 1)
                arg1 = ' '
            ps.update(text=ln)
            if c.startswith('@P'):
                ps.update(
                    tag='proof',
                    name='proof {:0>5d}'.format(stats.proofs),
                    topic=arg1,
                    authority=' ',
                )
                stats.proofs += 1
            elif c.startswith('@R'):
                ps.update(
                    tag='reality',
                    name='reality {:0>5d}'.format(stats.realities),
                    topic=arg1,
                    # proofs='proof 00000|',
                )
                stats.realities += 1
            elif c.startswith('@D'):
                ps.update(
                    tag='distortion',
                    name='dissent {:0>5d}'.format(stats.dissents),
                    topic=arg1,
                    # proofs='proof 00000|',
                )
                stats.dissents += 1
            elif c.startswith('@C'):
                ps.update(
                    tag='fraud',
                    name='conflict {:0>5d}'.format(stats.conflicts),
                    narrative=arg1,
                    # reality='reality 00000',
                    agency=' ',
                )
                stats.conflicts += 1
            elif c.startswith('@N'):
                ps.update(
                    tag='repeat',
                    conflict=' '  # 'fraud:conflict 00000',
                )
            elif c.startswith('@J'):
                ps.update(
                    tag='bias',
                    name='judgment {:0>5d}'.format(stats.judgments),
                    narrative=arg1,
                    # conflicts='fraud:conflict 00000|',
                    # dissents='distortion:dissent 00000|',
                    authority=' ',
                )
                stats.judgments += 1
            elif c.startswith('@A'):
                ps.update(
                    tag='perpetuate',
                    name='activism {:0>5d}'.format(stats.activisms),
                    narrative=arg1,
                    # judgments='bias:judgment 00000|',
                    authority=' ',
                )
                stats.activisms += 1
            else:
                assert c.startswith('@-')
                continue
            ps.update(**preset.get(c, {}))
        else:
            print('***', ln)
        yield ps
