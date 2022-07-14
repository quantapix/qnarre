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

from .counter import counters
from .contain import Contains


class Analyzer:

    san_args = ((('passed', '.'), ('failed', 'F')), 'Sanity:')

    def check_sanity(self, src, **kw):
        with counters(self.san_args, kw) as cs:
            return cs

    coh_args = ((('record', ''), ('purged', 'd'), ('equal', '='),
                 ('full', '<'), ('partial', '~')), 'Coherence:')

    def check_coherence(self, src, **kw):
        gs = Contains()
        with counters(self.coh_args, kw) as cs:
            gs.grow_from(src, **kw)
            mg, fg = gs.record, gs.full
            for m in sorted(mg.nodes()):
                if m in fg:
                    for m2 in fg.successors(m):
                        if m2 in fg and m in fg.successors(m2):
                            print(m, m2)
                            print(mg.node[m]['nominal'][:30],
                                  mg.node[m2]['nominal'][:30])
            return cs
