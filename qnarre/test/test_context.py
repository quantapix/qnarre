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

import pathlib as pth

from qnarre.error import ExcludeException
from qnarre.context import Adr, Adrs, Context


hdr1 = Adrs((Adr('aa', 'a@x.com'),
             Adr('bb', 'b@x.com')))
hdr2 = Adrs((Adr('cc', 'c@x.com'),
             Adr('aa', 'a2@x.com'),
             Adr('bb', 'b@x.com')))
hdr3 = Adrs((Adr('ae', 'a@x.com'),
             Adr('ee', 'e@x.com'),
             Adr('be', 'be@x.com')))

exc = ('e@x.com',)


def test_context_1():
    b = pth.Path('/tmp/qn-context')
    cp = b / 'context.qnr'
    if cp.exists():
        cp.unlink()
    c = Context.create(b, excluded=exc)
    c.slugs_for(hdr1)
    s = repr(sorted(c.contacts, key=lambda c: c.name))
    assert s == "[Contact('aa', ('a@x.com',)), Contact('bb', ('b@x.com',))]"
    c.slugs_for(hdr2)
    s = repr(sorted(c.contacts, key=lambda c: c.name))
    assert s == "[Contact('aa', ('a@x.com', 'a2@x.com')), Contact('bb', ('b@x.com',)), Contact('cc', ('c@x.com',))]"
    try:
        c.slugs_for(hdr3)
        assert False
    except ExcludeException:
        assert s == repr(sorted(c.contacts, key=lambda c: c.name))
    s = repr(c)
    assert s == "Context({'aa': Contact('aa', ('a@x.com', 'a2@x.com')), 'bb': Contact('bb', ('b@x.com',)), 'cc': Contact('cc', ('c@x.com',))}, {'e@x.com'})"
    c.save()
    c2 = Context.create(b)
    assert c == c2
    assert s == repr(c2)
    assert s == cp.read_text()
    if cp.exists():
        cp.unlink()
