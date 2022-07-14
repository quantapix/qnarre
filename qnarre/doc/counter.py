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
import curses

import asyncio as aio
import contextlib as cl
import multiprocessing as mp
import multiprocessing.pool as pl
import multiprocessing.queues as qs

from queue import Empty

from .log import Logger
from .resource import Resource

log = Logger(__name__)

cpu_count = os.cpu_count()

PID = 'pid'

START = 'start:'
STOP = 'stop:'

CNTR = 'cntr'


class Queue(qs.Queue):

    _pid = None

    def __init__(self):
        super().__init__(ctx=mp.get_context())
        q = mp.Queue()
        for i in range(cpu_count):
            q.put((PID, i + 1))
        self.cmdq = q

    @property
    def pid(self):
        if self._pid is None:
            self._pid = 1
            k, self._pid = self.cmdq.get()
            assert k == PID
        return self._pid

    def put(self, src, msg):
        super().put((src, (self.pid, msg)))


class Counters(Resource):

    queue = None

    title = None
    stats = None
    done = None

    _lines = None

    def __init__(self,
                 fields,
                 title='Starting:',
                 stats='Stats:',
                 done='done',
                 **kw):
        super().__init__(**kw)
        self.fields = fields
        if title is not None:
            self.title = title
        if stats is not None:
            self.stats = stats
        if done is not None:
            self.done = done

    @property
    def name(self):
        return CNTR

    @property
    def lines(self):
        if self._lines is None:
            self._lines = {}
        return self._lines

    def post(self, *args):
        m = ' '.join(a for a in args if a is not None)
        if self.queue is not None:
            self.queue.put(self.name, m)
        else:
            s = '\n' if args[0] in (START, STOP) else ''
            m = s + m
            print(m, end='', flush=True)

    def start(self):
        self._elems = {(k if k else n): 0 for n, k in self.fields}
        # self.post(START, self.title)
        self.count = 0

    def retitle(self, txt=''):
        self.post(START, ' '.join((self.title, txt)))

    def incr(self, key=None, amount=1):
        k = self.fields[0][1] if key is None else key
        self[k] += amount
        if len(k) == 1:
            m = k if self.count % 100 else '\n{:0>6d} {}'.format(self.count, k)
            self.count += 1
            self.post(m)
        return True

    def stop(self):
        if self.stats is not None:
            f = '{}: {}'
            s = (f.format(n, self[k if k else n]) for n, k in self.fields)
            s = ', '.join(s)
            self.post(STOP, self.stats, s)
        else:
            self.post(STOP, self.done)

    def track(self, scr, pid, msg):
        r = False
        assert pid > 0
        i = 3 * (pid - 1)
        if msg.startswith(START):
            ln = msg[len(START) + 1:]
            ln += ' ' * (100 - len(ln))
        elif msg.startswith(STOP):
            i += 2
            ln = msg[len(STOP) + 1:]
            ln += ' ' * (100 - len(ln))
            r = True
        else:
            i += 1
            if msg.startswith('\n'):
                ln = self.lines[i] = msg[1:]
                ln += ' ' * 100
            else:
                self.lines[i] += msg
                ln = self.lines[i]
        my, mx = scr.getmaxyx()
        my = (my // 3) * 3
        y = i % my
        x = 110 * (i // my)
        if x < mx:
            scr.addstr(y, x, ln[:mx - x])
        scr.refresh()
        return r


@cl.contextmanager
def counters(args, kw):
    try:
        cs = kw[CNTR]
    except KeyError:
        kw[CNTR] = cs = Counters(*args)
    cs.start()
    yield cs
    cs.stop()


class Worker:
    def __init__(self, queue, *args):
        self.queue = queue
        for a in args:
            if a is not None:
                setattr(self, a.name, a)

    def activate(self):
        for v in vars(self):
            if v != 'queue':
                getattr(self, v).queue = self.queue
        return self

    def run(self, mth, tgt, args, **kw):
        kw.update({v: getattr(self, v) for v in vars(self) if v != 'queue'})
        with counters(None, kw):
            return mth(tgt, *args, **kw)


worker = None


def init_worker(wrk):
    global worker
    worker = wrk.activate()


def run_worker(*args, **kw):
    global worker
    return worker.run(*args, **kw)


class Pool(pl.Pool):
    def __init__(self,
                 cntr=None,
                 lgr=None,
                 processes=None,
                 initializer=None,
                 initargs=None,
                 *args,
                 loop=None,
                 **kw):
        processes = processes or cpu_count
        initializer = initializer or init_worker
        self.tracker = t = Worker(Queue(), cntr, lgr)
        initargs = initargs or t,
        super().__init__(processes, initializer, initargs, *args, **kw)
        self.loop = loop or aio.get_event_loop()
        self.futs = []

    def call_worker(self, mth, tgt, args, **kw):
        f = self.loop.create_future()

        def _cb(res):
            def safe_cb(res):
                f.set_result(res)

            self.loop.call_soon_threadsafe(safe_cb, res)

        def _ecb(exc):
            def safe_ecb(exc):
                f.set_exception(exc)

            self.loop.call_soon_threadsafe(safe_ecb, exc)

        self.apply_async(run_worker, (mth, tgt, args), kw, _cb, _ecb)
        self.futs.append(f)

    async def track_workers(self):
        scr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        t = self.tracker
        try:
            scr.erase()
            for i in range(len(self.futs)):
                while True:
                    try:
                        s, m = t.queue.get_nowait()
                        await aio.sleep(.001)
                    except Empty:
                        await aio.sleep(.01)
                    else:
                        if getattr(t, s).track(scr, *m):
                            break
        finally:
            curses.echo()
            curses.nocbreak()
            curses.endwin()

    def run(self):
        f = aio.gather(*self.futs, self.track_workers(), loop=self.loop)
        self.loop.run_until_complete(f)


if __name__ == "__main__":

    class aa:
        def aaa(self, tgt, limit, *, cntr):
            for i in range(limit[0]):
                cntr.incr('.')

    fields = ((('scanned', '.'), ('excluded', '-'), ('imported', '+'),
               ('failed', 'F')), '*Testing*:')
    ax = aa()
    p = Pool(Counters(*fields), processes=10)
    for a in ((500, ), (600, ), (700, ), (800, ), (900, ), (1000, )):
        p.call_worker(aa.aaa, ax, (None, a))
    p.run()
