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


class Justifier:
    def __init__(self, **kw):
        super().__init__(**kw)
        self.justs = [0] * 9
        self.offsets = [(0, 0, 0, 1, 1, 1, 1, 1, 1),
                        (0, -1, -2, 0, 0, 0, 1, 1, 1),
                        (0, -1, -2, 0, -1, -2, 0, 0, 0)]

    def init_justs(self, justs):
        for i in justs:
            i = i // 3
            os = self.offsets[i]
            if os:
                self.justs = [sum(x) for x in zip(self.justs, os)]
                self.offsets[i] = None

    def calc_just(self, justs):
        for i in justs:
            i = self.justs[i] + (i % 3)
            if i == 1:
                return 'justify-content-center'
            elif i > 1:
                return 'justify-content-end'
        return 'justify-content-start'
