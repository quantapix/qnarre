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


class Stats:
    def __init__(self,
                 proofs=0,
                 realities=0,
                 dissents=0,
                 conflicts=0,
                 judgments=0,
                 activisms=0,
                 **_):
        self.proofs = proofs
        self.realities = realities
        self.dissents = dissents
        self.conflicts = conflicts
        self.judgments = judgments
        self.activisms = activisms

    def __str__(self):
        f = 'Stats: proofs {}, realities {}, dissents {}, '
        f += 'conflicts {}, judgments {}, activisms {}'
        return f.format(
            self.proofs,
            self.realities,
            self.dissents,
            self.conflicts,
            self.judgments,
            self.activisms,
        )

    @property
    def as_tuple(self):
        return (
            self.proofs,
            self.realities,
            self.dissents,
            self.conflicts,
            self.judgments,
            self.activisms,
        )
