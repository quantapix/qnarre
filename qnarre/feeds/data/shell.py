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

import sys
import hashlib

import subprocess as sub


class Shell:
    def __init__(self, path):
        self.path = path
        self.trace = []

    def run(self, *cmds):
        c = ';'.join((c for c in cmds if c is not None))
        print('==>', c)
        r = sub.run(
            c,
            cwd=self.path,
            capture_output=True,
            shell=True,
            text=True,
        )
        self.trace.append(r)
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        return r

    def out(self, *cmds):
        return self.run(*cmds).stdout

    def sha256sum(self, path):
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(self.path / path, 'rb', buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        return h.hexdigest()
