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
import filecmp as fc
import pathlib as pth
import collections as co

from hashlib import blake2b

from .log import Logger
from .base import config
from .counter import counters
from .resource import Resource, resource, Names

log = Logger(__name__)


def calc_digest(path, *, base=None, **_):
    p = base / path if base else pth.Path(path)
    if p.exists():
        d, s = blake2b(digest_size=20), 0
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(65536), b''):
                s += len(b)
                d.update(b)
        assert s == p.stat().st_size
        return d.hexdigest(), s
    log.warning("Cant't digest nonexistent file {}", p)
    return None, None


class Entry(co.namedtuple('Entry', 'path digest size')):

    __slots__ = ()

    def __new__(cls, path, digest=None, size=None, **kw):
        if not digest:
            digest, size = calc_digest(path, **kw)
        return super().__new__(cls, path, digest, size)

    def __bool__(self):
        return bool(self.path and self.digest is not None
                    and self.size is not None)

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            d = self.digest
            return (d and d == other.digest and self.size == other.size)
        return NotImplemented

    def __repr__(self):
        s = "{}({!r}".format(type(self).__name__, str(self.path))
        d = self.digest
        if d:
            s += ", {!r}, {}".format(d, self.size)
        s += ")"
        return s

    def relative_to(self, path, base, **_):
        try:
            (base / self.path).relative_to(base / path)
        except ValueError:
            return False
        return True

    def check(self, **kw):
        d = self.digest
        if d:
            d2, s = calc_digest(self.path, **kw)
            if d2 == d and s == self.size:
                return True
            m = 'Mismatched digest for {}'
        else:
            m = 'No digest for {}'
        log.info(m, self.path)
        return False


def prune_dir(path, cntr=None, **_):
    with os.scandir(path) as es:
        for e in es:
            p = pth.Path(e.path)
            j = None
            if p.name.startswith('.'):
                if e.is_dir(follow_symlinks=False):
                    sh.rmtree(str(p))
                elif p.suffix != '.qnr':
                    p.unlink()
                log.info('Deleted {}', p)
                j = '-'
            elif e.is_dir(follow_symlinks=False):
                prune_dir(p, cntr)
                continue
            if cntr:
                cntr.incr(j)
    try:
        path.rmdir()
        log.info('Deleted {}', path)
        j = '-'
    except:
        j = None
    if cntr:
        cntr.incr(j)


class Roster(Resource):

    _res_path = '.roster.qnr'

    @classmethod
    def globals(cls):
        return globals()

    def __init__(self, entries=None, **kw):
        super().__init__(None, **kw)
        self._expels = []
        self._symlinks = []
        if entries:
            self.add_entry(entries)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, tuple(self.entries))

    def __str__(self):
        s = '{}:'.format(str(self.base))
        for e in self.entries:
            s += '\n{} {} {}'.format(str(e.path), str(e.digest), e.size)
        return s

    @property
    def entries(self):
        es = [e for e in self.values() if isinstance(e, Entry)]
        return sorted(es, key=lambda x: x.path)

    def adjust_kw(self, kw):
        def _adjust(key, default):
            v = kw.get(key)
            v = pth.Path(v) if v else default
            kw[key] = v

        _adjust('base', self.base)

    def entry_adder(self, entry, cntr, modify=False, expel=True, **kw):
        if isinstance(entry, Entry):
            assert entry
            p, d, s = entry
            k = d, s
            if p in self:
                ok = self[p]
                if k != ok:
                    if modify:
                        log.info('Modifying digest for {}', p)
                        del self[ok]
                        self[p] = k
                        self[k] = entry
                        cntr.incr(modify)
                        return
                    else:
                        log.warning('Digest mismatch for {}', p)
                cntr.incr()
            else:
                try:
                    o = self[k]
                except KeyError:
                    self[p] = k
                    self[k] = entry
                    yield p
                else:
                    log.info('Duplicates: {} and {}', o.path, p)
                    if expel:
                        self._expels.append((o, entry))
                    cntr.incr()
        else:
            for e in entry:
                yield from self.entry_adder(e, cntr, modify, expel, **kw)

    add_args = ((('scanned', '.'), ('added', '+')), 'Adding:')

    def add_entry(self, entry, **kw):
        with counters(self.add_args, kw) as cs:
            for _ in self.entry_adder(entry, **kw):
                cs.incr('+')
            return cs

    def path_adder(self, path, **kw):
        self.adjust_kw(kw)
        p = str(pth.Path(path).relative_to(kw['base']))
        yield from self.entry_adder(Entry(p, **kw), **kw)

    def walker(self, paths=(), **kw):
        for e in self.entries:
            if paths:
                for p in paths:
                    if e.relative_to(p, **kw):
                        break
                else:
                    continue
            yield e

    def scanner(self, root, cntr, **kw):
        def _paths(path):
            with os.scandir(path) as es:
                for e in es:
                    p = pth.Path(e.path)
                    if not p.name.startswith('.'):
                        if e.is_dir(follow_symlinks=False):
                            yield from _paths(p)
                            continue
                        elif e.is_file(follow_symlinks=False):
                            yield p
                            continue
                        elif e.is_symlink():
                            log.info('Symlink {}', p)
                            self._symlinks.append(p)
                        else:
                            log.info('Ignoring dir entry {}', p)
                    cntr.incr()

        if root.exists():
            for p in _paths(root):
                yield from self.path_adder(p, **kw, cntr=cntr)

    scan_args = ((('scanned', '.'), ('added', '+')), 'Scanning:')

    def scan(self, paths=(), **kw):
        self.adjust_kw(kw)
        b = kw['base']
        with counters(self.scan_args, kw) as cs:
            for p in paths or ('', ):
                for _ in self.scanner(b / p, **kw):
                    cs.incr('+')
            return cs

    rescan_args = ((('scanned', '.'), ('added', '+'), ('removed', '-'),
                    ('modified', 'm')), 'Rescanning:')

    def rescanner(self, paths, cntr, **kw):
        self.adjust_kw(kw)
        b = kw['base']
        es = [e for e in self.walker(paths, **kw) if not (b / e.path).exists()]
        for p, d, s in es:
            del self[p]
            del self[(d, s)]
            cntr.incr('-')
        self._expels = []
        for p in paths or ('', ):
            for p in self.scanner(b / p, **kw, cntr=cntr, modify='m'):
                yield p

    def rescan(self, paths=(), **kw):
        with counters(self.rescan_args, kw) as cs:
            for _ in self.rescanner(paths, **kw):
                cs.incr('+')
            return cs

    check_args = ((('passed', '.'), ('failed', 'F')), 'Checking:')

    def check(self, paths=(), **kw):
        self.adjust_kw(kw)
        with counters(self.check_args, kw) as cs:
            for e in self.walker(paths, **kw):
                cs.incr('.' if e.check(**kw) else 'F')
            return cs

    def check_ok(self, paths=(), **kw):
        return not self.check(paths, **kw)['F']

    def rename_path(self, src, dst, cntr, cntr_key=None, **_):
        if dst.exists():
            log.warning("Can't move/rename, destination exists {}", dst)
            cntr.incr('F')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            log.info('Moved/renamed {} to/as {}', src, dst)
            cntr.incr(cntr_key)

    expel_args = ((('scanned', '.'), ('expelled', 'e'), ('failed', 'F')),
                  'Expelling:')

    def expel(self, ebase=None, **kw):
        with counters(self.expel_args, kw) as cs:
            self.adjust_kw(kw)
            b = kw['base']
            for o, d in self._expels:
                op = b / o.path
                dp = b / d.path
                if fc.cmp(op, dp, shallow=False):
                    e = (ebase or (b.parent / 'expel')) / d.path
                    self.rename_path(dp, e, **kw, cntr_key='e')
                else:
                    log.error('Duplicates compare failed {}, {}', op, dp)
                    cs.incr('F')
            self._expels = []
            return cs

    def absorb_paths(self, paths=(), abase=None, **kw):
        self.adjust_kw(kw)
        b = kw['base']
        ab = abase or (b.parent / 'absorb')
        for p in paths or ('', ):
            p = ab / p
            if p.exists():
                yield b, ab, p

    absorb_args = ((('scanned', '.'), ('absorbed', 'a'), ('failed', 'F')),
                   'Absorbing:')

    def absorb(self, paths=(), abase=None, **kw):
        with counters(self.absorb_args, kw) as cs:
            kw['expel'] = False
            for b, ab, path in self.absorb_paths(paths, abase, **kw):
                for p in [p for p in self.scanner(path, **kw, base=ab)]:
                    self.rename_path(ab / p, b / p, **kw, cntr_key='a')
                prune_dir(path)
            return cs

    prune_args = ((('scanned', '.'), ('deleted', '-')), 'Pruning:')

    def prune(self, paths=(), abase=None, **kw):
        with counters(self.prune_args, kw) as cs:
            for _, ab, p in self.absorb_paths(paths, abase, **kw):
                prune_dir(p, **kw)
            return cs

    def namer(self, path, names, base, cntr, **_):
        p = str(path)
        if p not in names:
            if (base / path).exists():
                names[p] = np = p.lower().replace(' ', '-')
                cntr.incr('.' if p == np else 'n')
                path = path.parent
                if path.name:
                    self.namer(path, names, base, cntr)
            else:
                cntr.incr('F')

    names_args = ((('scanned', '.'), ('renamed', 'r'), ('normalized', 'n'),
                   ('failed', 'F')), 'Naming:')

    def names(self, paths=(), **kw):
        with counters(self.names_args, kw) as cs:
            self.adjust_kw(kw)
            with resource(Names.create(kw['base'])) as ns:
                ns.clear()
                for e in self.walker(paths, **kw):
                    self.namer(pth.Path(e.path), ns, **kw)
            return cs

    rename_args = ((('scanned', '.'), ('added', '+'), ('removed', '-'),
                    ('modified', 'm'), ('normalized', 'n'), ('renamed', 'r'),
                    ('failed', 'F')), 'Renaming:')

    def rename(self, paths=(), **kw):
        with counters(self.rename_args, kw) as cs:
            self.adjust_kw(kw)
            b = kw['base']
            with resource(Names.create(b)) as ns:
                if ns:
                    for e in self.walker(paths, **kw):
                        p = e.path
                        try:
                            d = b / ns.pop(p)
                        except KeyError:
                            cs.incr()
                            continue
                        self.rename_path(b / p, d, **kw, cntr_key='r')
                ps = paths or ('', )
                for o in sorted(ns.keys(), reverse=True):
                    d = b / ns.pop(o)
                    o = b / o
                    if o.exists() and o.is_dir():
                        for p in ps:
                            try:
                                o.relative_to(b / p)
                                break
                            except ValueError:
                                continue
                        else:
                            cs.incr()
                            continue
                        self.rename_path(o, d, **kw, cntr_key='r')
                    else:
                        cs.incr()
                for p in self.rescanner(paths, **kw):
                    self.namer(pth.Path(p), ns, **kw)
            return cs


if __name__ == '__main__':
    from .args import BArgs
    a = BArgs()
    a.add_argument('paths', nargs='*', help='Paths to follow')
    a.add_argument('-u', '--prune', action=a.st, help='Prune absorb dir')
    a.add_argument('-a', '--absorb', help='Path to absorb uniques from')
    a.add_argument('-x', '--rename', action=a.st, help='Rename files')
    a.add_argument('-R', '--rescan', action=a.st, help='Rescan base')
    a.add_argument('-s', '--scan', action=a.st, help='Scan base')
    a.add_argument('-e', '--expel', help='Path to expel duplicates to')
    a.add_argument('-c', '--check', action=a.st, help='Check all digests')
    a.add_argument('-n', '--names', action=a.st, help='Names of files')
    a = a.parse_args()
    r = Roster.create(a.base)
    if a.prune:
        abase = None if a.absorb is None or a.absorb == config.DEFAULT else a.absorb
        r.prune(a.paths, abase=abase)
    elif a.absorb:
        abase = None if a.absorb == config.DEFAULT else a.absorb
        r.absorb(a.paths, abase=abase)
    elif a.rename:
        r.rename(a.paths)
    else:
        if a.rescan:
            r.rescan(a.paths)
        elif a.scan:
            r.scan(a.paths)
        if a.expel:
            ebase = None if a.expel == config.DEFAULT else a.expel
            r.expel(ebase=ebase)
        if a.check:
            r.check_ok(a.paths)
        if a.names:
            r.names(a.paths)
    r.save()
