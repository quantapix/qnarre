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

import datetime as dt

from ..rectify import rectifier

from .author import Author
from .named import Named, Saved


class Genre(Named):
    pass


class Doc(Saved, Named):
    suff = '.txt'
    pages = None

    def __init__(self, genre=None, author=None, title=None, pages=None, **kw):
        super().__init__(**kw)
        if genre:
            self.genre = Genre.create(name=genre)
        if author:
            self.author = author
        if title:
            self.title = title
        if pages:
            self.pages = pages

    @property
    def factor(self):
        return self.genre.factor * super().factor

    @property
    def bias(self):
        return self.genre.bias + super().bias

    @property
    def date(self):
        s = self.name.split('/')[2]
        s = '-'.join(s.split('-')[:3])
        return dt.datetime.strptime(s, '%y-%m-%d').date()

    @property
    def props(self):
        return {
            'name': self.name,
            'genre': self.genre.name,
            'author': self.author,
            'title': self.title,
        }

    @property
    def fields(self):
        s = '{}.pdf'.format(self.name)
        fs = {'Date': self.date, 'Title': self.title, 'Source': s}
        fs.update(Author.create(name=self.author).fields)
        fs.update({'Type': self.tag, 'Genre': self.genre.name})
        return fs

    def from_text(self, txt, **_):
        txt = tuple(rectifier(txt))
        self.title = txt[0]
        txt = '\n'.join(txt[2:])
        self.pages = gs = []
        for g in txt.split('\n\n\n'):
            rs = []
            for r in g.split('\n\n'):
                rs.append(r.splitlines())
            gs.append(rs)

    def to_text(self, **_):
        txt = [self.title, '']
        for rs in self.pages:
            for ls in rs:
                txt.extend(ls)
                txt.append('')
            txt.append('')
        return '\n'.join(txt).strip()
