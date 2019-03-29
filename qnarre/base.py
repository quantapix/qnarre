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
import re
import sys

import pathlib as pth
import collections as co

from hashlib import blake2b


def num_to_name(n):
    return '{:0>3d}0'.format(n)


def digest(value):
    return blake2b(value, digest_size=20).hexdigest()


def rst_def(pref, name):
    return '\n.. _{0}/{1}:\n\n{1}\n{2}\n'.format(pref, name, '=' * len(name))


def rst_ref(pref, name):
    return ':ref:`{}/{}`'.format(pref, name)


def lister(path, rng=(), suffs=('.png', '.jpg', '.mov')):
    with os.scandir(path) as es:
        for e in es:
            p = pth.Path(e.path)
            if p.is_file():
                if p.suffix in suffs and (not rng or p.stem in rng):
                    yield p
            elif p.is_dir():
                yield from lister(p, rng, suffs)


Adr = co.namedtuple('Adr', 'display_name addr_spec')
Adr.__new__.__defaults__ = ('', )


class Adrs(co.namedtuple('Adrs', 'addresses')):

    adr_pat = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    adr_re = re.compile(r'(?aim)' + adr_pat)

    @classmethod
    def has_adr(cls, txt):
        return bool(cls.adr_re.search(txt))

    @classmethod
    def from_txt(cls, txt):
        s = co.OrderedDict()
        txt = txt.replace(';', ',')
        for c in (', MD', ', Md', ',M.D.', ' M.D.', "'", '"', '*', '&',
                  'esquire', 'Esquire'):
            txt = txt.replace(c, ' ')
        t = txt
        for c in (',', '<', '[', '(', '>', ']', ')', 'mailto:'):
            t = t.replace(c, ' ')
        for w in t.split():
            if cls.adr_re.match(w):
                s[w] = None
        if s:
            return cls(tuple(Adr(None, a) for a in s.keys()))
        return (','.join(w for w in txt.split(',') if w.strip()), None)


def camelize(txt, upper_first=True):
    if upper_first:
        return re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), txt)
    else:
        return txt[0].lower() + camelize(txt)[1:]


def link_class(label):
    n = 'lnk_' + label
    n = camelize(n[:-1] if n.endswith('_') else n)
    d = dict(label=label, directed=label.endswith('ing'))
    globals()[n] = t = type(n, (object, ), d)
    return t


for l in ('full', 'partial'):
    link_class(l)

ls = (
    'audience',
    'bcc',
    'cc',
    'date',
    'from_',
    'including',
    'record_id',
    'proximity',
    'quoting',
    'referring',
    'replying',
    'source',
    'subject',
    'summary',
    'tags',
    'title',
    'to',
    'topic',
)

Hdr = co.namedtuple('Hdr', ls)
Hdr.links = tuple(link_class(l) for l in Hdr._fields)


class Record:

    label = 'record'


Traits = co.namedtuple('Traits', 'role background justify slug')
Traits.__new__.__defaults__ = (None, None, 0, None)


class Config:

    EQ = 'eq'
    LT = 'lt'
    GT = 'gt'

    TBD = 'TBD'
    DEFAULT = 'default'
    EXCLUDED = 'excluded'

    ENH = '_enh'

    HTML = 'html'
    ATTM = 'attm'
    PLAIN = 'plain'

    CHAIN = 'chain'

    def_from = ''

    include_adrs = ()
    exclude_specs = exclude_mids = ()
    exclude_doms = exclude_locs = exclude_fulls = ()

    ROOT = 'root'
    PRIV = 'priv'
    PROT = 'prot'
    PUBL = 'publ'
    OPEN = 'open'

    subject_aliases = topic_aliases = ()
    def_contacts = contact_aliases = bridge_aliases = {
        None: (),
        ROOT: (),
        PRIV: (),
        PROT: (),
        PUBL: (),
        OPEN: ()
    }

    SRC = 'src/'
    DST = 'dst/'

    CTXT = 'ctxt'
    DOCS = 'docs'
    PICS = 'pics'
    RECS = 'recs'

    ARCH = '/arch/'
    REPO = '/repo/'
    QNAR = 'qnar/'
    SAFE = '/safe/'
    BLOG = '/blog/'
    MAIN = '/main/'

    MBOX = 'mbox'
    TBOX = 'tbox'
    BBOX = 'bbox'
    # SBOX = 'transcripts'
    SBOX = 'try'
    IMGS = 'imgs'
    ORGS = 'orgs'

    nominal_offs = book_names = ()
    line_junk = line_replace = fixups = quotes = ()

    alloweds = substitutes = all_traits = {}

    web_templates = ''

    # Base RGB FFC0C0, Hue 0, Dist 90, Lightest Pale Pastel
    gray = 'e8e8e8'
    green = 'B8F4B8'
    lgreen = 'E4FDE4'
    blue = 'B7D0EC'
    lblue = 'E4EFFB'
    red = 'FFC0C0'
    lred = 'FFE6E6'
    yellow = 'FFF4C0'
    lyellow = 'FFFBE6'

    right = 8
    lright = right - 3
    middle = 7
    lmiddle = middle - 3
    left = 6
    lleft = left - 3

    @property
    def recs_src(self):
        return self.SRC + self.RECS

    @property
    def recs_arch(self):
        return self.SRC + self.RECS + self.ARCH

    @property
    def recs_repo(self):
        return self.SRC + self.RECS + self.REPO

    @property
    def main_src(self):
        return self.SRC + self.DOCS + self.MAIN

    @property
    def blog_src(self):
        return self.SRC + self.DOCS + self.BLOG

    @property
    def priv_src(self):
        return self.SRC + self.DOCS + self.SAFE

    @property
    def docs_src(self):
        return self.SRC + self.DOCS

    @property
    def sbox_src(self):
        return self.SRC + self.DOCS + self.REPO + self.SBOX

    @property
    def mbox_src(self):
        return self.recs_repo + self.MBOX

    @property
    def tbox_src(self):
        return self.recs_repo + self.TBOX

    @property
    def bbox_src(self):
        return self.recs_repo + self.BBOX

    @property
    def qnar_dst(self):
        return self.DST + self.QNAR

    @property
    def html_dst(self):
        return self.DST + self.QNAR + self.HTML

    @property
    def attm_dst(self):
        return self.DST + self.QNAR + self.ATTM


config = Config()

sys.path.insert(0, str(pth.Path.cwd()))
try:
    import qnarre_settings
    qnarre_settings.apply_to(config)
except ImportError as e:
    print('Failed to import a qnarre_settings.py', e)
sys.path.pop(0)


def traits_for(key):
    ts = config.all_traits.get(str(key), Traits())
    if ts.slug:
        ts = config.all_traits.get(ts.slug, Traits())
    return ts
