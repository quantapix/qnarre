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

import os

import shutil as sh
import filecmp as fl
import pathlib as pth

from hashlib import blake2b

from .log import Logger
from .date import Date
from .base import config, num_to_name
from .resource import Resource

log = Logger(__name__)

PDF = '.pdf'
DIG = '.dig'
RST = '.rst'


def digest(path):
    d, s = blake2b(digest_size=20), 0
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(65536), b''):
            s += len(b)
            d.update(b)
    assert s == path.stat().st_size
    return d.hexdigest()


def digester(path, names=((), ())):
    incs, excs = names
    with os.scandir(path) as es:
        for e in es:
            p = pth.Path(e.path)
            if p.is_file():
                yield p, digest(p)
            elif p.is_dir():
                n = p.name
                if n in incs or (not incs and n not in excs):
                    yield from digester(p, names)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(src, pth.Path):
        if dst.exists():
            assert fl.cmp(str(src), str(dst), False)
        else:
            sh.copy2(str(src), str(dst))
    else:
        assert isinstance(src, str)
        if dst.exists():
            assert src == dst.read_text()
        else:
            dst.write_text(src)


class Entry:
    def __init__(self, base, *args):
        _, p, np, self.seqn = args
        self.path = p.relative_to(base)
        self.npath = None if np is None else np.relative_to(base)
        self.dpaths = None

    def digest(self, base):
        return digest(base / self.path)

    def reflect_arch(self, oarch, narch, dig, suffs):
        np = self.npath
        np = narch / (self.path if np is None else np)
        s = np.suffix
        if self.seqn and (not suffs or s in suffs):
            np = np.with_name(num_to_name(self.seqn)).with_suffix(s)
        copy(oarch / self.path, np)
        if not suffs or s in suffs:
            assert dig == digest(np)

    def reflect_repo(self, nrepo, dig, suffs):
        np = self.npath
        np = nrepo / (self.path if np is None else np)
        s = np.suffix
        if self.seqn and (not suffs or s in suffs):
            np = np.with_name(num_to_name(self.seqn))
        op, ss = self.dpaths
        for e, s in ss:
            sp = op.with_suffix(s)
            dp = np.with_suffix(s)
            if e:
                sp = op.with_name(op.stem + e).with_suffix(s)
                if s == PDF:
                    dp = np.with_name(np.stem + e).with_suffix(s)
                    if RST not in ss:
                        pass
            copy(sp, dp)
        if DIG not in ss:
            copy(dig, np.with_suffix(DIG))

    def project(self, narch, nrepo, dig, suffs):
        p = self.npath
        p = self.path if p is None else p
        s = p.suffix
        if self.seqn and (not suffs or s in suffs):
            p = p.with_name(num_to_name(self.seqn)).with_suffix(s)
            copy(narch / p, nrepo / p)

    def reflect(self, oarch, narch, nrepo, dig, suffs):
        self.reflect_arch(oarch, narch, dig, suffs)
        if self.dpaths:
            self.reflect_repo(nrepo, dig, suffs)
        else:
            self.project(narch, nrepo, dig, suffs)


class Mirror(Resource):

    _res_path = 'mirror.qnr'

    @classmethod
    def globals(cls):
        return globals()

    def __init__(self,
                 elems=None,
                 arch='arch',
                 repo='repo',
                 names=((), ()),
                 ends=(config.ENH, ),
                 suffs=(),
                 **kw):
        super().__init__(elems, **kw)
        self.arch = arch
        self.repo = repo
        self.names = names
        self.ends = ends
        self.suffs = suffs

    def scan_arch(self):
        b = self.base / self.arch
        ss = self.suffs
        for r in Date.scanner(b, self.names, ss):
            e = Entry(b, *r)
            if not ss or e.path.suffix in ss:
                d = e.digest(b)
                if d in self:
                    oe = self[d]
                    if e.path != oe.path:
                        log.warning('Duplicates {} and {}', e.path, oe.path)
                else:
                    self[d] = e
            else:
                self[e] = e

    def scanner(self, path):
        nss = {}
        incs, excs = self.names
        with os.scandir(path) as es:
            for e in es:
                p = pth.Path(e.path)
                if p.is_file():
                    n = p.stem
                    for d in self.ends:
                        if n.endswith(d):
                            s = d, p.suffix
                            nss.setdefault(n[:-len(d)], []).append(s)
                            break
                    else:
                        s = None, p.suffix
                        nss.setdefault(n, []).append(s)
                elif p.is_dir():
                    n = p.name
                    if n in incs or (not incs and n not in excs):
                        yield from self.scanner(p)
                    else:
                        log.warning('Skipping dir {}', n)

        for n, ss in nss.items():
            yield path / n, tuple(ss)

    def scan_repo(self):
        b = self.base / self.repo
        for p, ss in self.scanner(b):
            d = None
            for e, s in ss:
                nd = None
                if s == DIG:
                    nd = p.with_suffix(s).read_text()
                elif e is None and s == PDF:
                    nd = digest(p.with_suffix(s))
                if nd is not None:
                    if d is None:
                        d = nd
                    else:
                        assert d == nd
            if d is None:
                log.warning('No digest anchor for {}', p)
                continue
            if d not in self:
                if not self.names[0]:
                    log.warning('No archive for {}', p)
            else:
                e = self[d]
                if e.dpaths:
                    op, oss = e.dpaths
                    log.warning('Duplicates {} and {}', str(op), str(p))
                else:
                    self[d].dpaths = p, ss

    def load(self):
        self.scan_arch()
        self.scan_repo()
        return self

    def reflect(self, narch='new-arch', nrepo='new-repo'):
        oarch = self.base / self.arch
        narch = self.base / narch
        nrepo = self.base / nrepo
        for d, e in self.items():
            e.reflect(oarch, narch, nrepo, d, self.suffs)
        if not self.names[0]:
            old = sorted(d for _, d in digester(oarch, self.names))
            new = sorted(d for _, d in digester(narch, self.names))
            assert old == new
            orepo = self.base / self.repo
            old = {d for _, d in digester(orepo, self.names)}
            new = {d for _, d in digester(nrepo, self.names)}
            assert old <= new
            print('... completed, checked')
        return self


if __name__ == '__main__':
    from .args import BArgs
    a = BArgs()
    a = a.parse_args()
    ns = ((), ())
    Mirror.create(a.base, names=ns).load().reflect()
