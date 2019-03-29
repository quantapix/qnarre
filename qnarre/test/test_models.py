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

import random as r

from django.utils.lorem_ipsum import sentence, paragraphs, words

from .realm import Realm, Subject, Agent
from .registry import Texts, Labels, Slugs
from .base import Title, Summary, Tag, Topic, Setting
from .section import Note, StoryPost, BlogPost, Doc, Story, Blog, Docs, Session


tis = tuple(
    Title('Ti' + str(i) + ' ' + words(1 + i, common=False).title())
    for i in range(7))

sus = tuple(
    Summary('Su' + str(i) + ' ' + sentence().capitalize())
    for i in range(5))

tps = tuple(
    Topic('Tp' + str(i) + ' ' + words(2, common=False).title())
    for i in range(10))

sbs = []
for i in range(30):
    t = 'Sb' + str(i) + ' ' + words(r.randint(2, 6), common=False).title()
    sbs.append(Subject(t[:15], text=t))

sbs_t = []
for i in range(30):
    t = 'Sb' + str(i) + ' ' + words(r.randint(2, 6), common=False).title()
    sbs_t.append(Subject(t[:15], topic=r.choice(tps), text=t))

ags = []
for i in range(10):
    fn = words(3, common=False)
    n, _ = fn.split(' ', 1)
    ags.append(Agent('Ag' + str(i) + ' ' + n, party=fn.title()))
for a in ags:
    a.party.email = a.slug + '@abc.com'

tgs = tuple(
    Tag('Tg' + str(i) + ' ' + words(1, common=False))
    for i in range(10))

notes = []
for i in range(100):
    d = '17-04-' + str(i)
    notes.append(Note(d, body=paragraphs(r.randint(1, 9), common=False)))
for n in notes:
    n.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))

sposts = []
sposts.append(
    StoryPost('17-01-01',
              body=paragraphs(r.randint(1, 9), common=False),
              subject=r.choice(sbs),
              title=r.choice(tis),
              summary=r.choice(sus)))
for i in range(10):
    sposts.append(
        StoryPost('17-01-1' + str(i),
                  body=paragraphs(r.randint(1, 9), common=False),
                  subject=r.choice(sbs) if r.randint(1, 4) != 2 else None,
                  title=r.choice(tis) if r.randint(1, 4) != 2 else None,
                  summary=r.choice(sus) if r.randint(1, 8) == 2 else None))
for p in sposts:
    p.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))

bposts = []
bposts.append(
    BlogPost('17-02-01',
             body=paragraphs(r.randint(1, 9), common=False),
             subject=r.choice(sbs_t),
             title=r.choice(tis),
             summary=r.choice(sus)))
for i in range(100):
    bposts.append(
        BlogPost('17-02-1' + str(i),
                 body=paragraphs(r.randint(1, 9), common=False),
                 subject=r.choice(sbs_t) if r.randint(1, 4) != 2 else None,
                 title=r.choice(tis) if r.randint(1, 4) != 2 else None,
                 summary=r.choice(sus) if r.randint(1, 2) == 2 else None))
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
                             'Sr' + str(i) + '-src-' + str(3 + i % 4 * 10))))
for d in docs:
    d.tags = (r.choice(tgs) for i in range(r.randint(1, 5)))
    d.froms = (r.choice(ags) for i in range(r.randint(1, 2)))
    d.tos = (r.choice(ags) for i in range(r.randint(1, 4)))

settings = tuple(Setting('St' + str(i) + ' ' + words(1, common=False),
                         value=words(1, common=False))
                 for i in range(10))

the_story = Story('My Story', sposts)
the_blog = Blog('My Blog', bposts)
the_docs = Docs('My Docs', (*notes, *docs))

the_session = Session('Buba Session', settings=settings, docs=the_docs)


if __name__ == '__main__':

    print('***Titles:')
    for t in tis:
        print(t)
    print('***Summaries:')
    for s in sus:
        print(s)
    print('***Tags:')
    for t in tgs:
        print(t, t.slug)
    print('***Topics:')
    for t in tps:
        print(t, t.slug)
    print('***Realm:', Realm.current)
    print('***Subjects:')
    for s in sbs:
        print(s.label, s, s.slug)
    print('***Agents:')
    for a in ags:
        print(a, a.slug, a.party, a.party.email)
    print('***Notes:')
    for n in notes:
        print(n, Tag.stringify_all(n.tags, sort=True))
        print(n.body)
        print('-------------')
    print('***Docs:')
    for d in docs:
        print(d)
        print(d.title)
        print(d.summary)
        print(d.subject)
        print(d.stringify_all(d.sources))
        print(d.stringify_all(d.tags, sort=True))
        print('Froms:', d.stringify_all(a.party for a in d.froms))
        print('Tos:', d.stringify_all(a.party for a in d.tos))
        print()
        print(d.body)
        print('============')
    print('***Settings:')
    for s in settings:
        print(s, s.slug, s.value)
    print('***', the_docs.title, '***')
    print('#topics', len(the_docs.topics))
    print('#subjects', len(the_docs.subjects))
    print('#items', len(the_docs.items))
    print('***', the_session.name, '***')
    print('#settings', len(the_session.settings))
    print('#story items', len(the_story.items))
    print('#blog items', len(the_blog.items))
    print('#texts', len(Texts.current))
    print('#labels', len(Labels.current))
    print('#slugs', len(Slugs.current))
