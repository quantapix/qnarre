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
import codecs

import datetime as dt
import collections as col


def _handler(err):
    c_map = {
        b'\x87': '',
        b'\xa3': '',
        b'\xa5': '',
        b'\xb4': '',
        b'\xb5': '',
        b'\xbc': '',
        b'\xc1': '',
        b'\xc7': '',
        b'\xc9': '',
        b'\xd2': '"',
        b'\xd3': '"',
        b'\xd4': "'",
        b'\xd5': "'",
        b'\xde': 'fi',
        b'\xdf': 'fl',
        b'\xe1': '',
    }
    k = err.object[err.start:err.end]
    # print('***', k, k.hex())
    if k in c_map:
        # print('replacing {} with {}'.format(k, c_map[k]))
        return c_map[k], err.end
    print(err.object[err.start - 20:err.end + 20])
    raise err


QNERR = 'qnerr'
codecs.register_error(QNERR, _handler)

flags = r'(?aim)'

mos = r'January|February|March|April|May|June|July|'
mos += r'August|September|October|November|December'

dp = r'(?P<dt>(?:' + mos + r') \d{1,2}, 20\d{2}),?'
dp = re.compile(flags + dp)


def days(txt):
    for p in dp.split(txt):
        for f in ('%B %d, %Y', ):
            try:
                d = dt.datetime.strptime(p, f)
                p = '{0:%y-%m-%d}'.format(d)
                break
            except ValueError:
                continue
        yield p


mp = r'(?P<mo>(?:' + mos + r') of 20\d{2})'
mp = re.compile(flags + mp)


def months(txt):
    for p in mp.split(txt):
        for f in ('%B of %Y', ):
            try:
                d = dt.datetime.strptime(p, f)
                p = '{0:%Y-%m}'.format(d)
                break
            except ValueError:
                continue
        yield p


s_map = col.OrderedDict()
s_map.update((
    ('Dr.', 'Dr '),
    ('Mr.', 'Mr '),
    ('Ms.', 'Ms '),
    ('Ofc.', 'Ofc '),
    ('Atty.', 'Atty '),
    ('Guardian ad litem', 'GAL'),
    ('Guardian ad Litem', 'GAL'),
    ('Guardian Ad Litem', 'GAL'),
    ('Department of Children and Family', 'DCF'),
    ('Department of Children and Families', 'DCF'),
    ('Concord Police Officers', 'Police'),
    ('Concord Police officers', 'Police'),
    ('Concord police officers', 'Police'),
    ('police officers', 'Police'),
    ('Concord Police Department', 'Police'),
    ('Concord District Court', 'District Court'),
    ('New Hampshire', 'NH'),
    ('the Commonwealth of Massachusetts', 'MA'),
    ('Massachusetts', 'MA'),
    ('Middlesex Probate and Family Court', 'Family Court'),
    ('Middlesex Probate & Family Court', 'Family Court'),
    ('Middlesex Division of the Probate and Family Court', 'Family Court'),
))
s_map.update((
    ('The Father', 'Dad'),
    ('the Father', 'Dad'),
    ('Father', 'Dad'),
    ('Imre Kifor', 'Dad'),
    ('Imre', 'Dad'),
    ('Barbara A.', 'Mom-B'),
    ('Barbara A', 'Mom-B'),
    ('Barbara', 'Mom-B'),
    ('Duchesne', 'Mom-B'),
    ('Ms Mom-B', 'Mom-B'),
    ('his former girlfriend', 'Mom-C'),
    ('Cynthia S.', 'Mom-C'),
    ('Cynthia S', 'Mom-C'),
    ('Cynthia', 'Mom-C'),
    ('Cyndi', 'Mom-C'),
    ('Cindy', 'Mom-C'),
    ('Oulton', 'Mom-C'),
    ('Ms Mom-C', 'Mom-C'),
    ('Twins', 'Kids-B'),
    ('twins', 'Kids-B'),
    ('Evan Kifor', 'Leon'),
    ('Evan', 'Leon'),
    ('Anna Kifor', 'Lisa'),
    ('Anna', 'Lisa'),
    ('Blake', 'Luke'),
    ('Belle', 'Lola'),
    ('Leon and Lisa', 'Kids-B'),
    ('Lisa and Leon', 'Kids-B'),
))
s_map.update((
    ('The Defendant', 'Dad'),
    ('the Defendant', 'Dad'),
    ('Defendant', 'Dad'),
    ('The Plaintiff', 'Mom-B'),
    ('the Plaintiff', 'Mom-B'),
    ('Plaintiff', 'Mom-B'),
    ('The children', 'children'),
    ('the children', 'children'),
    ('Children', 'Kids-B'),
    ('children', 'Kids-B'),
    ('Katie L. Lenihan, Esquire', 'Atty Lenihan'),
    ('Honorable Court', 'Court-B'),
    ('Sandy Mahoney', 'Ms Mahoney'),
))
s_map.update((
    ('Dad, Dad', 'Dad'),
    ('Dad Dad', 'Dad'),
    ('Mom-B, Mom-B', 'Mom-B'),
    ('Mom-B Mom-B', 'Mom-B'),
    ('Mom-C, Mom-C', 'Mom-C'),
    ('Mom-C Mom-C', 'Mom-C'),
    ('Kids-B, Kids-B', 'Kids-B'),
    ('Kids-B Kids-B', 'Kids-B'),
))
s_map.update((
    ('\r', '\n'),
    ('\t', ' '),
    (':', ' '),
    (';', ','),
    ('   ', ' '),
    ('  ', ' '),
    (' \n', '\n'),
    # ('?', '?.'),
    # ('!', '!.'),
    ('..', '.'),
    ('.\n', '\n'),
    (',\n', '\n'),
    ('. ', '\n'),
))


def rectify(txt):
    # txt = ''.join(days(txt))
    # txt = ''.join(months(txt))
    # for k, v in s_map.items():
    #     txt = txt.replace(k, v)
    return txt.strip()


def rectifier(txt):
    for ln in txt.splitlines():
        yield rectify(ln)


if __name__ == '__main__':
    print(rectify('dsfaf sc casdf Febru 25, 2011, dfwec asef ef'))
    print(rectify('dsfaf sc casdf February 25, 2011, dfwec asef ef'))
    print(rectify('dsfaf February 25, 2011, dfwec June 1, 2009 ef'))
