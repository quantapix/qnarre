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

import time

import asyncio as aio
import pathlib as pth
import contextlib as cl
import subprocess as sub

from aiohttp import ClientSession

from data.utils import Tokenizer


class Tokenizer(Tokenizer):

    headers = {'content-type': 'text/plain'}
    url = 'http://localhost:9000'
    params = {}

    def __init__(self, service, **kw):
        super().__init__(**kw)
        self.service = service

    def __call__(self, topics):
        self.count = 0
        loop = aio.get_event_loop()
        with server(self.service, loop):
            loop.run_until_complete(self._do_topics(topics, loop=loop))

    async def _do_topics(self, topics, loop):
        async with ClientSession(headers=self.headers) as s:
            for t in topics:
                print('Tokenizing topic: ' + t.title)
                for c in t.contexts:
                    c, ts = await self._do_context(c, s, loop)
                    c.tokens.elems = ts[0]
                    for i, qt in enumerate(ts[1:]):
                        q = c.questions[i]
                        q.tokens.elems = qt

    async def _do_context(self, ctxt, sess, loop):
        fs = [self._do_text(ctxt.text, sess)]
        fs.extend(self._do_text(q.text, sess) for q in ctxt.questions)
        print('... context of', len(fs))
        assert len(fs) < self.limit
        while self.count + len(fs) > self.limit:
            print('SLEEPING...')
            await aio.sleep(1)
        return ctxt, await aio.gather(*fs, loop=loop)

    async def _do_text(self, text, sess):
        d = text.replace('\n', ' ').encode()
        self.count += 1
        try:
            async with sess.post(self.url, data=d, **self.params) as r:
                return tuple(self.from_blob(await r.read()))
        finally:
            self.count -= 1


@cl.contextmanager
def server(service, loop=None):
    service.start(loop or aio.get_event_loop())
    yield
    service.stop()


class Service:

    server = None

    def __init__(self):
        n = type(self).__module__.lower()
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
            cs = self.command.split()
            n = aio.subprocess.DEVNULL
            s.child = loop.run_until_complete(
                # aio.create_subprocess_exec(*cs, loop=loop))
                aio.create_subprocess_exec(*cs, loop=loop, stdout=n, stderr=n))
            loop.run_until_complete(aio.sleep(1, loop=loop))
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


class Server:

    count = 1
    child = None


@cl.contextmanager
def locked(path, parents=True, timeout=None):
    pl = PathLock(path, parents, timeout)
    pl.acquire()
    yield
    pl.release()


class PathLock:

    locked = False

    def __init__(self, path, parents=True, timeout=None):
        if parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path if path.name == 'lock' else path.with_suffix('.lock')
        self.timeout = timeout or 10

    def acquire(self):
        assert not self.locked
        for _ in range(self.timeout):
            try:
                self.path.touch(exist_ok=False)
                # print('Locked:', str(self.path))
            except FileExistsError:
                time.sleep(.5)
            else:
                self.locked = True
                return
        raise TimeoutError('Lock acquire failed')

    def release(self):
        assert self.locked
        self.path.unlink()
        del self.locked
