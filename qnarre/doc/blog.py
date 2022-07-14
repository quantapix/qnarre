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

import shutil as sh
import filecmp as fl
import pathlib as pth

from .log import Logger
from .base import config
# from .mirror import copy
from .counter import counters

log = Logger(__name__)

SUFF = '.rst'


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


class Blog:
    def __init__(self, base):
        self.base = base

    populate_args = ((('chained', '.'), ('blogged', '+'), ('excluded', '-'),
                      ('failed', 'F')), 'Populating:')

    def populate(self, dst, ctxt, **kw):
        kw.update(ctxt=ctxt)
        dst = self.base / dst
        with counters(self.export_args, kw) as cs:
            for _, ms in ctxt.recs.chainer(**kw):
                for m in ms:
                    a = '\n'.join(m.blogger(**kw))
                    (dst / m.slug).with_suffix(SUFF).write_text(a)
                    o = m.hdr.original
                    if o:
                        s = self.base / config.docs_src / m.source / o
                        copy(s, dst / s.name)
                    cs.incr('+')
            return cs
