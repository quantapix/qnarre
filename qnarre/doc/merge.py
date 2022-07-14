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

from qnarre import load_from, load_docs


def merge(root, genres=None, authors=None, **kw):
    kw.update(root=root, preset={})
    print('Loading from {}...'.format(str(root)))
    ds = [d for d in load_docs(genres=genres, authors=authors, **kw)]
    n = load_from(pth.Path('merged.py'), **kw)
    print('...done')
    print('Merging ({} + 1)...'.format(len(ds)))
    n.org.docs = sorted(ds, key=lambda d: d.date)
    n.org.save()
    print('...done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('-r', '--root', help='Path to root', default=None)
    args.add_argument('-g', '--genre', help='Genre to load', default=None)
    args.add_argument('-a', '--author', help='Author to load', default=None)
    args = args.parse_args()
    kw = {}
    if args.genre:
        kw['genres'] = args.genre,
    if args.author:
        kw['authors'] = args.author,
    merge(pth.Path.cwd() / (args.root or 'sample'), **kw)
