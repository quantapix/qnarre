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

from .log import Logger
from .header import Header

log = Logger(__name__)


class PackHeader(Header):
    def __init__(self, hdr, **kw):
        super().__init__({}, **kw)
        self.extract(vars(hdr))

    def merge(self, other):
        super().merge(other)
        if not self.subject and other.subject:
            self.subject = other.subject
