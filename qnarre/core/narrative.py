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

from ..edit import fudge

from .named import Named


class Topic(Named):
    pass


class Node(Named):
    _topic = _narrative = None

    def __init__(self, topic=None, narrative=None, **kw):
        super().__init__(**kw)
        topic = fudge(2) if topic == 'fudge' else topic
        if topic:
            self._topic = Topic.create(name=topic)
        narrative = fudge(2) if narrative == 'fudge' else narrative
        if narrative:
            self._narrative = Narrative.create(name=narrative)

    def __str__(self):
        t = '({} {}) {}'
        return t.format(self.sign, self.name, self.value)

    @property
    def topic(self):
        n = self.narrative
        return self._topic or (n.topic if n else n)

    @property
    def narrative(self):
        return self._narrative

    @property
    def value(self):
        t = self.topic.name if self.topic else ''
        n = self.narrative.name if self.narrative else ''
        return '[{}:{}]'.format(t, n)

    @property
    def fields(self):
        fs = super().fields
        fs['Topic'] = self.topic.name if self.topic else None
        fs['Narrative'] = self.narrative.name if self.narrative else None
        return fs


class Narrative(Node):
    sign = ''
