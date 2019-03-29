# Copyright 2018 Quantapix Authors. All Rights Reserved.
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

import lzma

import pathlib as pth

from qfeeds.prep.utils import normalize, Embedder


class Embedder(Embedder):

    base = pth.Path.home() / 'cache/qmodels/glove/raw'

    def __init__(self, src, words=None, limit=None):
        p = self.base / src
        self.words = dict(self.loader(p, words, limit))
        p = p.with_name(p.stem + '-char').with_suffix(p.suffix)
        self.chars = dict(self.loader(p, limit))

    def loader(self, path, words=None, limit=None):
        with open(path, 'rt') as f:
            for i, ln in enumerate(f):
                if limit and i > limit:
                    break
                ln = ln.rstrip().split(' ')
                w = normalize(ln[0])
                if words is None:
                    yield w, tuple(float(f) for f in ln[1:])
                else:
                    if w in words:
                        yield words[w], tuple(float(f) for f in ln[1:])

    def load_embed(self, path, limit=None):
        d = {}
        with lzma.open(path, 'rt') as f:
            for i, ln in enumerate(f):
                if limit and i > limit:
                    break
                ps = ln.rstrip().split(' ')
                d[self.words[ps[0]]] = tuple(float(i) for i in ps[1:])
        return d
