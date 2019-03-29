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

import shutil
import logging

import asyncio as aio
import pathlib as pth
import contextlib as cl
import subprocess as sub

from ...base import Base
from ...utils import locked, Marked, unmarked

log = logging.getLogger(__name__)

LOCK = 'lock'
MARK = 'mark'


class Server:

    count = 1
    child = None


class CoreNLP(Base):

    home = 'data/corenlp'
    latest = 'stanford-corenlp-full-2018-02-27'

    _files = pth.Path('/tmp') / Base.home_path.name / home
    _sources = _files / 'edu/stanford/nlp'

    server = None
    server_pid = _files / 'server_pid.txt'

    def __init__(self, root=None, **kw):
        super().__init__(root=root, **kw)
        if not tuple(self.home_path.glob('*.xz')):
            self.download()

    def download(self, name=None, archive=False):
        h = self.home_path
        name = self.latest if name is None else name
        if not (h / (name + '.tar.xz')).exists():
            c = str(h)
            if not (h / (name + '.tar')).exists():
                for f in self.home_path.glob('*.tar'):
                    sub.run('rm (!s}'.format(f).split(), cwd=c)
                n = name + '.zip'
                u = 'http://nlp.stanford.edu/software/'
                sub.run('wget -O {} {}'.format(n, u + n).split(), cwd=c)
                sub.run('unzip {}'.format(n).split(), cwd=c)
                sub.run('rm {}'.format(n).split(), cwd=c)
                self.clean()
            if archive:
                for f in self.home_path.glob('*.xz'):
                    sub.run('rm {!s}'.format(f).split(), cwd=c)
                sub.run('xz -v -9 -T0 {}'.format(name + '.tar').split(), cwd=c)

    @property
    def files(self):
        fs = self._files
        with locked(fs / LOCK):
            with cl.suppress(Marked), unmarked(fs / MARK):
                h = self.home_path
                g = tuple(h.glob('*.xz'))
                c = str(h)
                if g:
                    xz = sub.Popen(
                        'xz -dc {!s}'.format(g[0]).split(),
                        stdout=sub.PIPE,
                        cwd=c)
                    tar = 'tar xf - --strip-components=1 -C {!s}'
                    tar = sub.run(
                        tar.format(fs).split(),
                        stdin=xz.stdout,
                        stdout=sub.PIPE,
                        cwd=c)
                    xz.stdout.close()
                    xz.wait()
                else:
                    f = next(h.glob('*.tar'))
                    tar = 'tar xf {!s} --strip-components=1 -C {!s}'
                    sub.run(tar.format(f, fs).split(), cwd=c)
        return fs

    @property
    def sources(self):
        fs, ss = self.files, self._sources
        with locked(fs / LOCK):
            with cl.suppress(Marked), unmarked(ss / MARK):
                j = next(fs.glob('stanford-corenlp-*-sources.jar'))
                sub.run('jar xf {!s}'.format(j).split(), cwd=str(fs))
        return ss

    def clean(self, path=None):
        fs, ss = self._files, self._sources
        with locked(fs / LOCK):
            p = fs if path is None else path
            if p == fs or p == ss:
                if p.exists():
                    shutil.rmtree(p)
                self.protoc(clean=True)

    def protoc(self, update=False, clean=False):
        h = self.home_path
        n = 'CoreNLP.proto'
        p = h / 'CoreNLP_pb2.py'
        if not p.exists() or update:
            s = self.sources / 'pipeline'
            c = 'protoc -I={!s}/ --python_out={!s} {}'
            sub.run(c.format(s, h, n).split())
            shutil.copy(s / n, h)
        elif clean:
            p.unlink()
            (h / n).unlink()

    def start(self, loop=None):
        if loop is None or self.server is None:
            fs = self.files
            with locked(fs / LOCK):
                if loop:
                    self.server = Server()
                p = self.server_pid
                if not p.exists():
                    loop = loop or aio.get_event_loop()
                    cs = ('/usr/bin/java -mx8g -cp {!s}/* ' +
                          'edu.stanford.nlp.pipeline.StanfordCoreNLPServer ' +
                          '-port 9000 -timeout 15000')
                    cs = cs.format(fs).split()
                    n = aio.subprocess.DEVNULL
                    s = loop.run_until_complete(
                        aio.create_subprocess_exec(
                            *cs, loop=loop, stdout=n, stderr=n))
                    loop.run_until_complete(aio.sleep(.05, loop=loop))
                    p.write_text(str(s.pid))
                    if self.server:
                        self.server.child = s
        else:
            self.server.count += 1

    def stop(self):
        if self.server:
            self.server.count -= 1
            if self.server.count:
                return
        fs = self._files
        with locked(fs / LOCK):
            p = self.server_pid
            if p.exists():
                if self.server is None:
                    sub.run('kill {}'.format(p.read_text()).split())
                else:
                    self.server.child.kill()
                    del self.server
                p.unlink()
            assert self.server is None

    def sentences_from(self, buf, offset=0):
        self.protoc()
        from .CoreNLP_pb2 import Document
        d = Document()
        from google.protobuf.internal.decoder import _DecodeVarint
        size, pos = _DecodeVarint(buf, offset)
        d.ParseFromString(buf[offset + pos:offset + pos + size])
        return d.sentence


@cl.contextmanager
def server(corenlp, loop=None):
    corenlp.start(loop or aio.get_event_loop())
    yield
    corenlp.stop()
