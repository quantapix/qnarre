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

from docutils import statemachine
from docutils.parsers.rst import Directive, directives

from .log import Logger
from .dispatch import Dispatch

log = Logger(__name__)


class Excerpt(Directive):

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'inline': directives.flag,
        'start-line',
        'end-line',
        'start-after': directives.unchanged_required,
        'end-before': directives.unchanged_required
    }

    _dispatch = None

    @property
    def dispatch(self):
        if self._dispatch is None:
            type(self)._dispatch = Dispatch.create()
        return self._dispatch

    def run(self):
        sm = self.state_machine
        s = sm.input_lines.source(self.lineno - sm.input_offset - 1)
        p = self.arguments[0]
        self.state.document.settings.record_dependencies.add(p)
        kw = dict(
            start_line=self.options.get('start-line', None),
            end_line=self.options.get('end-line', None),
            start_after=self.options.get('start-after', None),
            end_before=self.options.get('end-before', None))
        t = self.dispatch.excerpt(s, p, **kw)
        ls = statemachine.string2lines(t, convert_whitespace=True)
        sm.insert_input(ls, p)
        return []


def setup(app):
    app.add_directive('excerpt', Excerpt)
    return {'version': '0.1'}
