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

from qnarre import load_from


def spread(root, **kw):
    kw.update(root=root)
    print('Loading {}...'.format(str(root)))
    o = load_from(pth.Path('merged.org'), **kw)
    print('...done')
    print('Spreading ({} + 1)...'.format(len(o.docs)))
    for d in o.docs:
        d.save(**kw)
    o.net.save(**kw)
    print('...done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('-r', '--root', help='Path to root', default=None)
    args = args.parse_args()
    spread(pth.Path.cwd() / (args.root or 'sample'))
