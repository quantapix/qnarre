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

# import asyncio as aio

# from aiohttp import ClientSession
# from async_timeout import timeout

from qnarre.feeds.prep.encoder import BertEncoder

# from qnarre.feeds.prep.corenlp import CoreNLP, server


class Tokenizer:
    def __init__(self, params):
        self.PS = PS = params
        self.encoder = BertEncoder.load(PS)

    def __call__(self, items):
        for i in items:
            yield self.encode(i)

    def encode(self, item):
        if hasattr(item, 'text') and hasattr(item, 'tokens'):
            print(item.text[:20], len(item.text))
            item.tokens.reset(*zip(*self.encoder(item.text)))
            print(len(item.tokens.elems), len(item.tokens.offsets))
        elif isinstance(item, list) or isinstance(item, tuple):
            for i in item:
                self.encode(i)
        return item

    def decode(self, ids, offsets):
        return self.encoder.decode(ids, offsets)

    def test(self, txt):
        self.encoder.test(txt)


"""
class BatchTokenize:

    timeout = 5000
    url = 'http://localhost:9000'
    annos = 'tokenize ssplit lemma pos ner'.split()
    # annos = 'tokenize ssplit lemma pos ner depparse'.split()
    headers = {'content-type': 'text/plain; charset=utf-8'}
    serializer = 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'

    def __init__(self, hook=None, annos=None, limit=500):
        self.intern_hook = hook
        if annos is not None:
            self.annos = annos
        self.limit = limit
        self.post_kw = {
            'params': {
                'properties':
                str({
                    'inputFormat': 'text',
                    'outputFormat': 'serialized',
                    'serializer': self.serializer,
                    'annotators': ','.join(self.annos)
                })
            }
        }
        self.corenlp = CoreNLP()

    def __call__(self, topics, loop=None):
        with server(self.corenlp, loop):
            self.count = 0
            loop = loop or aio.get_event_loop()
            for t in topics:
                yield loop.run_until_complete(self.from_topic(t, loop=loop))

    async def from_topic(self, topic, loop):
        print('Tokenizing topic: ' + topic.title)
        fs = [self.from_context(c, loop) for c in topic.contexts]
        while fs:
            done, fs = await aio.wait(
                fs, loop=loop, return_when=aio.FIRST_COMPLETED)
            for c in done:
                c, ts = c.result()
                c.tokens._data = ts[0]
                for i, t in enumerate(ts[1:]):
                    q = c.questions[i]
                    q.tokens._data = t
        return topic

    async def from_context(self, context, loop):
        fs = [self.from_text(context.text)]
        fs.extend(self.from_text(q.text) for q in context.questions)
        assert len(fs) < self.limit
        while self.count + len(fs) > self.limit:
            print('Sleeping... {}/{}'.format(self.count, self.limit))
            await aio.sleep(1)
        return context, await aio.gather(*fs, loop=loop)

    async def from_text(self, text):
        try:
            self.count += 1
            d = text.encode('utf-8')
            async with ClientSession(headers=self.headers) as s:
                with timeout(self.timeout):
                    async with s.post(self.url, data=d, **self.post_kw) as r:
                        b = await r.read()
                        return tuple(self.from_buf(b))
        finally:
            self.count -= 1

    def from_buf(self, buf):
        ts = [t for s in self.corenlp.sentences_from(buf) for t in s.token]
        for i, t in enumerate(ts):
            w = t.word
            fix = {
                '-LRB-': '(',
                '-RRB-': ')',
                '-LSB-': '[',
                '-RSB-': ']',
                '-LCB-': '{',
                '-RCB-': '}'
            }
            s = Span(t.beginChar, t.endChar)
            w, p, m, n = fix.get(w, w), t.pos, t.lemma, t.ner
            h = self.intern_hook
            if h:
                w, p, m, n = h(w), h(p), h(m), h(n)
            yield Token(word=w, span=s, pos=p, lemma=m, ner=n)
"""
