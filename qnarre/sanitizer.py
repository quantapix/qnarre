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

import gzip
import codecs

import pathlib as pth

from qnarre.log import Logger

log = Logger(__name__)

c_map = {
    '\u2013': '-',
    '\xe9': 'e',
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\xbd': 'half',
    '\x96': '"',
    '\u2014': '-',
    '\u2018': "'",
    '\u2026': '...',
    '\xb8': ',',
    '\u2022': '-',
    '\xa7': 'para. ',
    '\xa9': '(c)',
    '\xae': '(R)',
    '\x92': "'",
    '\x93': '"',
    '\x94': '"',
    '\x99': '-',
    '\xad\xad': '',
    '\ufffd': 'ee',
    '\u2122': '(TM)',
    '””': '"',
    'ï¿½': '',
    '™”': '"',
    '\u200e': '',
    '—“': '- "',
    '▶': '',
    '”’': '"-s',
    '��': '',
    '…”': '... "',
    '😊': ':-)',
    '😎': ';-)',
    '›': '',
    '“…': '"...',
    '”…': '"...',
    'ü': 'u',
    '😳': '',
    '😭': '',
    '😴': '',
    '😂': '',
    '😉': ';-)',
    'ó': 'o',
    'é’': "e'",
    '\u200b': '',
    '••••••••': '.....',
    '“…”': '"..."',
    '😢': ':-(',
    '———————————————————————————————————————————————————————': '',
    '·': '',
    'Â©': '(c)'
}

# '\xd4': "'", 'd5': "'", 'd2': '"', 'd3': '"',
# 'de': 'fi', 'df': 'fl', 'a5': 'M', 'e1': '?',
# 'a2': '?', 'db': '?'

s_map = {'\r': '\n', '\t': ' ', '  ': ' ', ' \n': '\n'}


def qnarre_handler(err):
    # k = err.object[err.start:err.end].hex()
    k = err.object[err.start:err.end]
    if k in c_map:
        # print('replacing {} with {}'.format(k, c_map[k]))
        return c_map[k], err.end
    # print(err.object[err.start - 20:err.end + 20])
    raise err


QNERR = 'qnerr'

codecs.register_error(QNERR, qnarre_handler)


def sanitize(txt):
    if isinstance(txt, str):
        txt = txt.replace('\xa0', ' ')
        try:
            return txt.encode('ascii', QNERR).decode('ascii', QNERR)
        except UnicodeError:
            # print(repr(txt))
            raise
    elif isinstance(txt, pth.Path):
        p = txt
        s = p.suffix
        t = p.with_suffix('.qpx')

        def _sanitize(o):
            with o(t, 'w+t', encoding='ascii', errors=QNERR) as d:
                with o(p, 'rt') as s:
                    for ln in s:
                        ln = ln.encode('ascii', QNERR)
                        d.write(ln.decode('ascii', QNERR))

        if s == '.gz':
            _sanitize(gzip.open)
        else:
            _sanitize(open)
        t.rename(p)
    elif txt:
        print('sanitize called on', repr(txt))
    return txt


class Sanitizer:

    base = None

    @classmethod
    def create(cls, base=None):
        if base:
            cls.base = pth.Path(base)
        return cls()

    def load(self, path):
        def _text_at():
            b = self.base
            p = b / path if b else pth.Path(path)
            try:
                s = p.read_text(errors='qnarre')
            except UnicodeDecodeError as e:
                log.error('Decode error {}', e)
                raise e
            for k, v in s_map.items():
                s = s.replace(k, v)
            return p, s

        self._path, self._text = _text_at()

    def dump(self, path=None):
        p = path or self._path
        p.write_text(self._text)


if __name__ == '__main__':
    import argparse as ap
    a = ap.ArgumentParser()
    a.add_argument('files', nargs='*', help='Files to read')
    a = a.parse_args()
    c = pth.Path.cwd()
    for f in a.files:
        sanitize(c / f)
