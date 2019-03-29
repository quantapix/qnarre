# Copyright 2018 Quantapix Authors. All Rights Reserved.
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

# import pytest

import shutil as sh
import pathlib as pth

from qnarre.context import Adr, Adrs, Context
from qnarre.record import Record
from qnarre.recs import Recs


hdr = Adrs((Adr('A. Bcd', 'abcd@gmail.com'),
            Adr('B. Cde', 'bcde@gmail.com'),
            Adr('C. Def', 'cdef@gmail.com'),
            ))

excluded = ()

used_keys = ('message-id', 'in-reply-to', 'date', 'subject',
             'from', 'to', 'cc', 'bcc')

ignore_keys = ('authentication-results', 'content-transfer-encoding',
               'delivered-to', 'dkim-signature', 'domainkey-signature',
               'received', 'received-spf', 'return-path')

x_keys = ('x-aol-global-disposition', 'x-aol-sid', 'x-mailer',
          'x-mb-message-source', 'x-mb-message-type', 'x-originating-ip',
          'x-proofpoint-spam-details', 'x-proofpoint-virus-version',
          'x-received', 'x-yahoo-newman-id', 'x-yahoo-newman-property',
          'x-yahoo-smtp', 'x-ymail-osg')

rest_keys = ('content-type', 'mime-version', 'references')


def test_message_1():
    b = pth.Path('/tmp/qn-message')
    b.mkdir(parents=True, exist_ok=True)
    sh.copy(str(pth.Path.cwd() / 'test.mbox'), str(b))
    cp = b / 'context.qnr'
    if cp.exists():
        cp.unlink()
    c = Context.create(b, excluded=excluded)
    c.slugs_for(hdr)
    c.save()
    s = repr(sorted(c.contacts, key=lambda c: c.name))
    ms = Recs.create(b)
    ir = ms.eml_importer(b, ('test'), ctxt=c)
    for m in ir:
        continue
        print(repr(m))
        print(m.name)
        print('**********************************')
    assert s == repr(sorted(c.contacts, key=lambda c: c.name))
    c.save()
    ak = Message._all_keys
    ak.difference_update(used_keys, ignore_keys, x_keys)
    assert ak == set(rest_keys)


def test_message_2():
    b = pth.Path('/tmp/qn-message')
    b.mkdir(parents=True, exist_ok=True)
    sh.copy(str(pth.Path.cwd() / 'txt_test.pdf'), str(b))
    sh.copy(str(pth.Path.cwd() / 'txt_test.txt'), str(b))
    cp = b / 'context.qnr'
    if cp.exists():
        cp.unlink()
    c = Context.create(b, excluded=excluded)
    c.slugs_for(hdr)
    c.save()
    s = repr(sorted(c.contacts, key=lambda c: c.name))
    ms = Recs.create(b)
    ir = ms.txt_importer(b, ('txt_test.txt'), ctxt=c)
    for m in ir:
        continue
    assert s == repr(sorted(c.contacts, key=lambda c: c.name))
    c.save()
    ms.save()
    s2 = repr(ms)
    ms2 = Recs.create(b)
    ir = ms2.txt_importer(b, ('txt_test.pdf'), ctxt=c)
    for m in ir:
        continue
    assert s == repr(sorted(c.contacts, key=lambda c: c.name))
    c.save()
    ms2.save()
    assert s2 == repr(ms2)
    assert ms == ms2
