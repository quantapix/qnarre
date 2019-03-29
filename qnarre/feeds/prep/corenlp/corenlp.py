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

import logging

import asyncio as aio
import pathlib as pth
import contextlib as cl
import subprocess as sub

from aiohttp import ClientSession
from async_timeout import timeout

from ..utils import locked
from .token import Span, Token

log = logging.getLogger(__name__)


class Server:

    count = 1
    child = None


class Service:

    server = None

    def __init__(self):
        n = type(self).__name__.lower()
        self.pid = pth.Path('/tmp') / (n + '_service/pid.txt')
        self.lock = self.pid.parent / 'lock'

    def start(self, loop):
        s = self.server
        if s:
            s.count += 1
            return
        with locked(self.lock):
            assert not self.pid.exists()
            self.server = s = Server()
            cs = ('java -mx8g -cp {!s}/* '
                  'edu.stanford.nlp.pipeline.StanfordCoreNLPServer '
                  '-port 9000 -timeout 15000')
            cs = cs.format(self.class_path).split()
            n = aio.subprocess.DEVNULL
            s.child = loop.run_until_complete(
                aio.create_subprocess_exec(*cs, loop=loop, stdout=n, stderr=n))
            loop.run_until_complete(aio.sleep(.05, loop=loop))
            self.pid.write_text(str(s.child.pid))

    def stop(self):
        s = self.server
        if s:
            s.count -= 1
            if s.count:
                return
        with locked(self.lock):
            p = self.pid
            if p.exists():
                if s is None:
                    sub.run('kill {}'.format(p.read_text()).split())
                else:
                    s.child.kill()
                    del self.server
                p.unlink()
            assert self.server is None

    def sentences_from(self, buf, offset=0):
        from .corenlp_pb2 import Document
        d = Document()
        from google.protobuf.internal.decoder import _DecodeVarint
        size, pos = _DecodeVarint(buf, offset)
        d.ParseFromString(buf[offset + pos:offset + pos + size])
        return d.sentence


@cl.contextmanager
def server(service, loop=None):
    service.start(loop or aio.get_event_loop())
    yield
    service.stop()


class CoreNLP(Service):

    class_path = '/home/qpix/tools/corenlp'


class Tokenizer:

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
        log.info('Tokenizing topic: ' + topic.title)
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
            log.info('Sleeping... {}/{}'.format(self.count, self.limit))
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
