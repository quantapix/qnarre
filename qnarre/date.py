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

import re
import os

import pathlib as pth
import datetime as dt

from qnarre.log import Logger
from qnarre.base import config

log = Logger(__name__)


def slugify(name):
    return name.replace('|', '_').replace(':', '-')


class Date:

    date_re = re.compile(r'\d\d?:\d\d', re.ASCII)

    delta = dt.timedelta(hours=3, minutes=30)
    fudge = dt.timedelta(minutes=1)

    _name = None
    _raw = None

    @classmethod
    def has_date(cls, txt):
        return bool(cls.date_re.search(txt))

    @classmethod
    def create_from(cls, txt, fmts, time=True, force=False):
        t = txt.strip()
        for f in fmts:
            try:
                d = dt.datetime.strptime(t, f).astimezone()
                if not d.second and (force or
                                     (time and not d.hour and not d.minute)):
                    d = d.replace(second=1)
                return cls(d)
            except ValueError as e:
                err = e
        raise err

    @classmethod
    def from_txt(cls, txt):
        return cls.create_from(
            txt, (
                '%b %d, %Y %I!%M!%S %p',
                '%b %d, %Y %I:%M:%S %p',
                '%m/%d/%y, %I:%M %p',
                '%m/%d/%y, %I:%M:%S %p',
                '%b %d, %Y %H:%M:%S',
            ),
            force=True)

    @classmethod
    def from_inl(cls, txt):
        txt = txt.replace('*', '')
        return cls.create_from(txt, (
            '%b %d, %Y %I:%M %p',
            '%m/%d/%Y %I:%M %p',
            '%b %d, %Y, at %I:%M %p',
            '%b %d, %Y, at %H:%M',
            '%a, %b %d, %Y at %I:%M %p',
        ))

    @classmethod
    def from_fwd(cls, txt):
        txt = txt.replace('*', '')
        return cls.create_from(txt, (
            '%B %d, %Y at %I:%M:%S %p %Z',
            '%A, %B %d, %Y %I:%M %p',
            '%A, %B %d, %Y, %I:%M %p',
            '%A, %B %d, %Y %I:%M:%S %p',
            '%a, %b %d, %Y at %I:%M %p',
            '%a %m/%d/%Y %I:%M %p',
            '%m/%d/%Y %I:%M %p',
            '%a, %b %d, %Y at %I:%M:%S %p',
            '%a, %b %d, %Y %I:%M %p',
            '%a %b %d %H:%M:%S %Y',
            '%a %b %d %H:%M:%S %Z %Y',
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
            '%B %d, %Y %I:%M:%S %p %Z',
            '%B %d, %Y %I:%M %p',
            '%d %b \'%y %I:%M',
            '%a, %B %d, %Y %I:%M %p',
            '%B %d, %Y, %I:%M:%S %p %Z',
        ))

    @classmethod
    def from_pth(cls, txt):
        return cls.create_from(
            txt, (
                '%y-%m-%d',
                '%Y-%m-%d',
            ), time=False)

    @classmethod
    def from_file(cls, path):
        s = path.stat()
        try:
            t = s.st_birthtime
        except AttributeError:
            t = s.st_mtime
        return cls(dt.datetime.fromtimestamp(t).astimezone())

    @classmethod
    def scanner(cls, path, names=((), ()), suffs=(), date=None, npath=None):
        i = 0
        incs, excs = names
        with os.scandir(path) as es:
            for e in sorted(es, key=lambda e: e.name):
                p = pth.Path(e.path)
                old = p.stem
                try:
                    d = cls.from_pth(old)
                except ValueError:
                    d = date
                    new = None
                else:
                    assert date is None
                    new = d.short
                    new = None if new == old else new
                s = p.suffix
                if npath is not None or new is not None:
                    new = (npath or path) / (new or old)
                    new = new.with_suffix(s)
                if p.is_file():
                    if not suffs or s in suffs:
                        if d is date:
                            i += 1
                            yield d, p, new, i
                        else:
                            yield d, p, new, 0
                elif p.is_dir():
                    if old in incs or (not incs and old not in excs):
                        yield from cls.scanner(p, names, suffs, d, new)
                    else:
                        log.warning('Skipping dir {}', p.name)

    def __init__(self, spec):
        if isinstance(spec, str):
            self._name = spec
        else:
            assert isinstance(spec, dt.datetime)
            self._raw = spec

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.name)

    @property
    def name(self):
        if self._name is None:
            r = self._raw
            ms = r.microsecond
            if r.hour == r.minute == r.second == 0:
                self._name = '{0:%y-%m-%d}|{1:0>3d}'.format(r, ms)
            else:
                ms = ':{}'.format(ms) if ms else ''
                self._name = '{0:%y-%m-%d}|{0:%H:%M:%S}'.format(r) + ms
        return self._name

    @property
    def raw(self):
        if self._raw is None:
            n = self._name
            c = len(n.split(':'))
            if c == 4:
                f = '%y-%m-%d|%H:%M:%S:%f'
            elif c == 3:
                f = '%y-%m-%d|%H:%M:%S'
            else:
                assert c == 1
                f = '%y-%m-%d|%f'
            self._raw = dt.datetime.strptime(n, f).astimezone()
        return self._raw

    @property
    def short(self):
        return '{0:%y-%m-%d}'.format(self.raw)

    @property
    def proximity(self):
        return '{0:%y-%m-%d}'.format(self.raw - self.delta)

    @property
    def slug(self):
        return slugify(self.name)

    @property
    def micro(self):
        return self.raw.microsecond

    @micro.setter
    def micro(self, value):
        self._raw = self.raw.replace(microsecond=value)
        if self._name is not None:
            del self._name

    @property
    def zero_secs(self):
        r = self.raw
        f = '{0:%y-%m-%d}|{0:%H:%M:00}'
        n = f.format(r)
        if r.second:
            return n, f.format(r + self.fudge)
        return n,

    @property
    def to_inl(self):
        return '{0:%b %d, %Y, at %I:%M %p}'.format(self.raw)

    @property
    def to_rst(self):
        return '{0:%Y-%m-%d %H:%M:%S}'.format(self.raw)

    def compare(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        s = self.raw.replace(microsecond=0)
        o = other.raw.replace(microsecond=0)
        if s == o:
            return config.EQ
        if s == o.replace(second=0):
            return config.LT
        if o == s.replace(second=0):
            return config.GT
        if s == (o + self.fudge).replace(second=0):
            return config.LT
        if o == (s + self.fudge).replace(second=0):
            return config.GT

    def after(self, others):
        m = max(d.micro for d in others if self.compare(d) is config.EQ)
        self.micro = m + 1

    def next_hour(self, delta=1):
        return type(self)(self.raw + dt.timedelta(hours=delta))

    def next_sec(self, delta=1):
        return type(self)(self.raw + dt.timedelta(seconds=delta))
