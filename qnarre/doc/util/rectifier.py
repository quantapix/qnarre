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

import re
import pathlib as pth

from .sanitizer import QNERR


class Rectify:

    end_re = re.compile(r'(:\d\d) ?:', re.ASCII)

    def __init__(self, path):
        self.path = path
        self.txt = pth.Path(path).read_text('ascii', QNERR)

    def fix_up(self):
        t = self.txt
        t = ' '.join(t.split()).strip()
        t = self.end_re.sub(r'\1:\n\n', t)
        # t = t.replace(' :', ' :\n\n')
        t = t.replace(' .', '.')
        p = pth.Path(self.path)
        p = p.with_name('new_' + p.stem).with_suffix('.txt')
        p.write_text(t, 'ascii', QNERR)


if __name__ == '__main__':
    Rectify('test.txt').fix_up()
