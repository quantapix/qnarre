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

from .edit import resolve
from .nominals import para_split
from .justifier import Justifier
from .meta import converter, with_property
from .realm import About, Sent, Sourced, Body
from .part import Defaults, Titled, Dated, Tagged, Part


class Defaults(Defaults):

    title = None
    summary = None
    subject = None
    subgroup = None

    body = None


@converter('TxtRec')
@converter('ScrRec')
@with_property('body', Body.create)
class Note(Dated, Sent, Tagged, Defaults):

    parent = None
    no_date = False

    @classmethod
    def convert(cls, rec, regy, ctxt, **kw):
        kw.update(regy=regy)
        s = rec.slug
        try:
            n = regy[cls.slugify(s)]
        except KeyError:
            n = cls(label=s, **kw)
        n.convert_from(rec, **kw, ctxt=ctxt)
        return n

    def __init__(self, label, body=None, **kw):
        super().__init__(label, **kw)
        self.body = body

    def convert_from(self, rec, ctxt, **kw):
        self.body = Body.create(self.body)
        self.body.texts = para_split(resolve(rec.text(ctxt), ctxt))
        if rec.no_date:
            self.no_date = True
        super().convert_from(rec, **kw, ctxt=ctxt)

    @property
    def date(self):
        return '' if self.no_date else super().date

    @property
    def html(self):
        b = self.body
        return b.html if b else super().html

    @property
    def justify(self):
        if self.parent:
            return self.parent.calc_just(f.justify for f in self.froms)
        return 'justify-content-start'

    @property
    def text_justify(self):
        return ''
        # return 'text-right' if self.justify == 'justify-content-end' else ''

    @property
    def background(self):
        for f in self.froms:
            if f.background:
                return 'background-color: #{};'.format(f.background)
        return 'background-color: #e8e8e8;'


@converter('InlRec')
@converter('FwdRec')
@converter('EmlRec')
@converter('MixRec')
@with_property('replying', Note.create)
class Message(About, Note):
    def __init__(self, label, replying=None, **kw):
        super().__init__(label=label, **kw)
        self.replying = replying

    def convert_from(self, rec, ctxt, **kw):
        r = rec.hdr.replying
        if r:
            pass
            # self.replying = Message.convert(ctxt.recs[r], **kw)
        super().convert_from(rec, **kw, ctxt=ctxt)


@converter('Chain')
@with_property('notes', Note.creator)
class Chain(Part, Titled, About, Sent, Tagged, Justifier, Defaults):
    @classmethod
    def convert(cls, chain, regy, ctxt, **kw):
        kw.update(regy=regy)
        n = chain.name
        try:
            c = regy[cls.slugify(n)]
        except KeyError:
            c = cls(label=n, **kw)
        c.convert_from(chain, **kw, ctxt=ctxt)
        return c

    @classmethod
    def get_template(cls):
        return 'chain'

    def __init__(self, label, notes=(), **kw):
        super().__init__(label, **kw)
        self.notes = notes

    def convert_from(self, chain, ctxt, **kw):
        kw.update(ctxt=ctxt)
        rs = ctxt.recs
        self.notes = (converter.convert(rs[n], **kw) for n in chain.names)
        for n in self.notes:
            n.parent = self
            self.init_justs(f.justify for f in n.froms)
        super().convert_from(chain, **kw)


class Letter(Sourced, Message):
    pass


class Post(Titled, About, Note):
    pass


@converter('StoryRec')
class StoryPost(Post):
    @classmethod
    def get_template(cls):
        return 'story'


@converter('BlogRec')
class BlogPost(Post):
    @classmethod
    def get_template(cls):
        return 'blog'


@converter('PicRec')
@converter('DocRec')
class Doc(Titled, Sourced, Message):
    pass
