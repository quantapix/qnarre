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

import os

import pathlib as pth
import importlib as imp

__version__ = '0.2.0'


def to_tag(suff):
    return {
        '.boot': 'boot',
        '.org': 'org',
        '.preset': 'preset',
        '.py': 'net',
        '.txt': 'doc',
    }[suff]


def to_class(tag):
    m = imp.import_module('qnarre.core')
    return getattr(m, tag.capitalize())


def create_from(*, tag, name=None, **kw):
    kw.update(tag=tag)
    # c = globals()[tag.capitalize()]
    c = to_class(tag)
    if name is None:
        return c(**kw)
    return c.create(name=name, **kw)


def load_from(path, **kw):
    kw.update(path=path)
    t = to_tag(path.suffix)
    n = path.stem
    if t == 'doc':
        n = '{}s/{}/{}'.format(kw['genre'], kw['author'], n)
    print('{}:{}'.format(t, n))
    return create_from(tag=t, name=n, **kw)


def load_docs(*, root, genres=None, authors=None, preset=None, **kw):
    kw.update(root=root)
    if genres is None:
        from qnarre.core import all_genres
        genres = all_genres
    for g in genres:
        kw.update(genre=g)

        def scan_dir(path, **kw):
            if path.exists():
                with os.scandir(path) as es:
                    for e in es:
                        p = pth.Path(e.path)
                        if p.name.startswith('.') or p.name.startswith('_'):
                            continue
                        if p.is_dir():
                            a = p.stem
                            if authors is None or a in authors:
                                yield from scan_dir(p, author=a, **kw)
                        elif p.is_file():
                            if p.suffix == '.txt':
                                yield load_from(p.relative_to(root), **kw)
                            elif preset is not None and p.suffix == '.preset':
                                ps = load_from(p.relative_to(root), **kw)
                                preset.update(ps.props)

        yield from scan_dir(root / (g + 's'), **kw)
