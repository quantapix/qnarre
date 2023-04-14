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
import email

import pathlib as pth
import contextlib as cl

from mailbox import mbox
from email.message import EmailMessage as MboxEntry

from .log import Logger
from .base import config
from .reader import Reader
from .record import EmlRec
from .nominals import nominal
from .sanitizer import sanitize
from .counter import counters, Counters, Pool

log = Logger(__name__)


class Mbox(mbox):
    @staticmethod
    def qfactory(fp):
        from email import policy
        return email.message_from_binary_file(fp, policy=policy.default)

    @classmethod
    def qsrc(cls, path, sort_mbox=False, **_):
        b = cls(path)
        if sort_mbox:
            return sorted(b, key=lambda m: m['date'].datetime.astimezone())
        return b

    def __init__(self, path, factory=None, create=False):
        t = None
        if path.suffix in ('.xz', ):
            import tempfile
            t = tempfile.NamedTemporaryFile()
        super().__init__(
            t.name if t else path, factory=self.qfactory, create=create)
        if t:
            if create:
                self.qpath = path
            else:
                import lzma
                self._file = lzma.open(path)

    def close(self):
        if hasattr(self, 'qpath'):
            self._file.seek(0)
            import lzma
            with lzma.open(self.qpath, 'wb') as dst:
                import shutil
                shutil.copyfileobj(self._file, dst)
        super().close()


def qextract_text(self, dejunk=True):
    vs = []
    for p in self.walk():
        if p.get_content_type() == 'text/' + config.PLAIN:
            v = sanitize(p.get_content())
            if dejunk:
                v = EmlRec.junk.dejunk_text(v)
            if v:
                vs.append(v)
    return vs


assert not hasattr(MboxEntry, 'qextract_text')
MboxEntry.qextract_text = qextract_text


class MboxEntry(MboxEntry):
    @classmethod
    def qcreate(cls, src):
        i = cls()
        for k, v in src:
            if k in ('text/' + config.PLAIN, 'text/' + config.HTML):
                try:
                    v = sanitize(v)
                except UnicodeError:
                    v = 'UnicodeError'
                i.set_content(v, subtype=k.split('/')[1], cte='7bit')
            else:
                i[k] = v
        return i


class Writer:
    def __init__(self, path):
        self.path = path

    @cl.contextmanager
    def to_mbox(self):
        p = self.path
        if p.exists():
            p.unlink()
        mb = Mbox(p, create=True)
        yield mb
        mb.flush()
        mb.close()


SUFF = '.mbox'
PROT = config.PROT

ORIG = config.recs_src + '/' + config.MBOX
ARCH = config.recs_arch + config.MBOX
REPO = config.recs_repo + config.MBOX


class Mboxes:
    def __init__(self, base):
        self.base = base

    def filt_one(self, stem, cntr, **kw):
        n = stem + SUFF
        with Writer(self.base / ARCH / n).to_mbox() as mb:
            kw.update(files=(stem, ), cntr=cntr)
            for m, _, _, _ in EmlRec.filterer(self.base / ORIG, **kw):
                mb.add(m)
                cntr.incr('+')

    filt_args = ((('excluded', '-'), ('allowed', '+'), ('failed', 'F')),
                 'Filtering:')

    def filt_mbox(self, files, **kw):
        with counters(self.filt_args, kw):
            for f in files:
                self.filt_one(f, **kw)

    def pool_filt(self, **kw):
        pool = Pool(Counters(*self.filt_args))
        with os.scandir(self.base / ORIG) as es:
            for e in es:
                p = pth.Path(e.path)
                if p.is_file() and p.suffix == SUFF:
                    pool.call_worker(Mboxes.filt_one, self, (p.stem, ), **kw)
        pool.run()

    def merge_two(self, dst, src, wdir, cntr, **kw):
        assert dst != src
        kw.update(cntr=cntr)
        wdir = self.base / wdir
        ms = {}
        for _, m in EmlRec.reader(Reader(wdir), **kw, files=(dst, )):
            ms.setdefault(m['message-id'], []).append(m)

        def body(m):
            b = m.get_body(preferencelist=(config.PLAIN, ))
            return nominal(b.get_content()) if b else None

        dirty = False
        for _, m in EmlRec.reader(Reader(wdir), **kw, files=(src, )):
            mid = m['message-id']
            if mid in ms:
                b1 = body(m)
                ls = ms[mid]
                for m2 in ls:
                    if b1 == body(m2):
                        cntr.incr('-')
                        break
                else:
                    ls.append(m)
                    cntr.incr('+')
                    dirty = True
            else:
                ms[mid] = [m]
                cntr.incr('+')
                dirty = True
        (wdir / src).with_suffix(SUFF).unlink()
        if dirty:
            ms = (m for l in ms.values() for m in l)
            ms = sorted(ms, key=lambda m: m['date'].datetime.astimezone())
            with Writer((wdir / dst).with_suffix(SUFF)).to_mbox() as mb:
                mb.clear()
                for m in ms:
                    mb.add(m)

    merge_args = ((('added', '+'), ('skipped', '-'), ('failed', 'F')),
                  'Merging:')

    def merge_mbox(self, files, **kw):
        with counters(self.merge_args, kw):
            dirty = True
            while dirty:
                dirty = False
                prev = None
                fs = []
                for f in files:
                    if prev:
                        self.merge_two(prev, f, **kw)
                        fs.append(prev)
                        dirty = True
                        prev = None
                    else:
                        prev = f
                if dirty:
                    files = fs

    def pool_merge(self, files, wdir, **kw):
        kw.update(wdir=wdir)
        dirty = True
        while dirty:
            dirty = False
            prev = None
            pool = Pool(Counters(*self.merge_args))
            with os.scandir(self.base / wdir) as es:
                for e in es:
                    p = pth.Path(e.path)
                    if p.is_file() and p.stem != PROT and p.suffix == SUFF:
                        if prev:
                            pool.call_worker(Mboxes.merge_two, self,
                                             (prev.stem, p.stem), **kw)
                            dirty = True
                            prev = None
                        else:
                            prev = p
            if dirty:
                pool.run()

    def strip_one(self, stem, wdir, cntr, dejunk_only=None, **kw):
        n = stem + SUFF
        with Writer(self.base / REPO / n).to_mbox() as mb:
            kw.update(files=(stem, ), cntr=cntr)
            if dejunk_only:
                for _, m in EmlRec.reader(Reader(self.base / wdir), **kw):
                    try:
                        t = '\n'.join(m.qextract_text())  # dejunk=False
                    except UnicodeError:
                        cntr.incr('-')
                    else:
                        m.clear_content()
                        m.set_content(t, subtype=config.PLAIN, cte='7bit')
                        mb.add(m)
                        cntr.incr('+')
            else:
                for m in EmlRec.importer(self.base / wdir, **kw):
                    m.junk = False
                    mb.add(MboxEntry.qcreate(m.mboxer(**kw)))
                    cntr.incr('+')

    strip_args = ((('excluded', '-'), ('stripped', '+'), ('failed', 'F')),
                  'Stripping:')

    def strip_mbox(self, files, **kw):
        with counters(self.strip_args, kw):
            for f in files:
                self.strip_one(f, **kw)

    def pool_strip(self, files, wdir, **kw):
        kw.update(wdir=wdir)
        pool = Pool(Counters(*self.strip_args))
        with os.scandir(self.base / wdir) as es:
            for e in es:
                p = pth.Path(e.path)
                if p.is_file() and p.stem != PROT and p.suffix == SUFF:
                    pool.call_worker(Mboxes.strip_one, self, (p.stem, ), **kw)
        pool.run()

    export_args = ((('chained', '.'), ('exported', '+'), ('excluded', '-'),
                    ('failed', 'F')), 'Exporting:')

    def export_to(self, dst, ctxt, **kw):
        kw.update(ctxt=ctxt)
        dst = (self.base / dst).with_suffix(SUFF)
        with counters(self.export_args, kw) as cs:
            for t, ms in ctxt.recs.chainer(**kw):
                p = dst
                if t:
                    p = p.with_name(p.stem + '_' + t).with_suffix(p.suffix)
                with Writer(p).to_mbox() as mb:
                    for m in ms:
                        mb.add(MboxEntry.qcreate(m.mboxer(**kw)))
                        cs.incr('+')
            return cs
