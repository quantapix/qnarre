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

import pathlib as pth

from django.conf import settings
from django.apps import AppConfig

from ..part import Parts
from ..base import config
from ..realm import realms
from ..dispatch import Dispatch
from ..resource import resource
from ..section import Story, Blog, Agents


class QnarreConfig(AppConfig):

    name = 'qnarre'

    def ready(self):
        self.parts_all = ps = Parts()
        b = pth.Path(settings.BASE_DIR)
        s = getattr(settings, 'IMGS_SRC', config.DST)
        for r in realms:
            with resource(Dispatch.create(b, r)) as d:
                d.ctxt.imgs_src = s
                d.convert(ps)
        self.parts_flat = sorted(ps.values(), key=lambda p: p.name)
        self.story = Story(self.parts_flat)
        self.blog = Blog(self.parts_flat)
        self.agents = Agents(self.parts_flat)
