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
"""
import random as r

from django.utils.lorem_ipsum import sentence, paragraphs, words

from .realm import Subject, Agent
from .section import Story, Blog, Docs, Session
from .message import Note, StoryPost, BlogPost, Doc
from .part import Title, Summary, Tag, Topic, Setting, Parts


qs = Parts.current

tis = tuple(
    Title('Ti' + str(i) + ' ' + words(1 + i, common=False).title())
    for i in range(7))

sus = tuple(
    Summary('Su' + str(i) + ' ' + sentence().capitalize())
    for i in range(5))

tps = tuple(
    Topic('Tp' + str(i) + ' ' + words(2, common=False).title(), regy=qs)
    for i in range(10))

sbs = []
for i in range(30):
    t = 'Sb' + str(i) + ' ' + words(r.randint(2, 6), common=False).title()
    sbs.append(Subject(t[:15], text=t, regy=qs))

sbs_t = []
for i in range(30):
    t = 'Sb' + str(i) + ' ' + words(r.randint(2, 6), common=False).title()
    sbs_t.append(Subject(t[:15], topic=r.choice(tps), text=t, regy=qs))

ags = []
for i in range(10):
    fn = words(3, common=False)
    n, _ = fn.split(' ', 1)
    ags.append(Agent('Ag' + str(i) + ' ' + n, contact=fn.title(), regy=qs))
for a in ags:
    a.contact.www = a.slug + '@abc.com'

tgs = tuple(
    Tag('Tg' + str(i) + ' ' + words(1, common=False), regy=qs)
    for i in range(10))

notes = []
for i in range(100):
    d = '17-04-' + str(i)
    notes.append(Note(d, body=paragraphs(r.randint(1, 9), common=False),
                      regy=qs))
for n in notes:
    n.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))

sposts = []
sposts.append(
    StoryPost('17-01-01',
              body=paragraphs(r.randint(1, 9), common=False),
              subject=r.choice(sbs),
              title=r.choice(tis),
              summary=r.choice(sus),
              regy=qs))
for i in range(10):
    sposts.append(
        StoryPost('17-01-1' + str(i),
                  body=paragraphs(r.randint(1, 9), common=False),
                  subject=r.choice(sbs) if r.randint(1, 4) != 2 else None,
                  title=r.choice(tis) if r.randint(1, 4) != 2 else None,
                  summary=r.choice(sus) if r.randint(1, 8) == 2 else None,
                  regy=qs))
for p in sposts:
    p.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))

bposts = []
bposts.append(
    BlogPost('17-02-01',
             body=paragraphs(r.randint(1, 9), common=False),
             subject=r.choice(sbs_t),
             title=r.choice(tis),
             summary=r.choice(sus),
             regy=qs))
for i in range(100):
    bposts.append(
        BlogPost('17-02-1' + str(i),
                 body=paragraphs(r.randint(1, 9), common=False),
                 subject=r.choice(sbs_t) if r.randint(1, 4) != 2 else None,
                 title=r.choice(tis) if r.randint(1, 4) != 2 else None,
                 summary=r.choice(sus) if r.randint(1, 2) == 2 else None,
                 regy=qs))
for p in bposts:
    p.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))

docs = []
for i in range(100):
    docs.append(Doc('17-03-1' + str(i),
                    body=paragraphs(1 + i % 7, common=False),
                    subject=r.choice(sbs_t) if r.randint(1, 4) != 2 else None,
                    title=r.choice(tis) if r.randint(1, 4) != 2 else None,
                    summary=r.choice(sus) if r.randint(1, 8) == 2 else None,
                    sources=('Sr' + str(i) + '-src-' + str(i % 5),
                             'Sr' + str(i) + '-src-' + str(i % 3),
                             'Sr' + str(i) + '-src-' + str(3 + i % 4 * 10)),
                    regy=qs))
for d in docs:
    d.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))
    d.froms = (r.choice(ags) for i in range(r.randint(1, 2)))
    d.tos = (r.choice(ags) for i in range(r.randint(1, 4)))

settings = tuple(Setting('St' + str(i) + ' ' + words(1, common=False),
                         value=words(1, common=False),
                         regy=qs)
                 for i in range(10))

the_story = Story('My Story', sposts)
the_blog = Blog('My Blog', bposts)
the_docs = Docs('My Docs', (*notes, *docs))

the_session = Session('Buba Session', settings=settings, docs=the_docs)
"""
