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

import pathlib as pth
import argparse as ap

from .base import config
from .log import Logger, start_stop_log

log = Logger(__name__)


def default_base():
    b = pth.Path.cwd()
    s = b / config.SRC
    assert s.exists() and s.is_dir()
    d = b / config.DST
    assert d.exists() and d.is_dir()
    # q = b / config.QPY
    # assert q.exists() and q.is_dir()
    return b


class Namespace(ap.Namespace):
    @property
    def base(self):
        b = self.basepath
        b = pth.Path(b) if b else default_base()
        assert b.exists() and b.is_dir()
        return b

    @property
    def kw(self):
        return self.__dict__


class BArgs(ap.ArgumentParser):

    st = 'store_true'

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_argument('-b', '--basepath', help='Path to base')

    def parse_args(self, args=None):
        return super().parse_args(args, Namespace())


class CArgs(BArgs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        cs = (config.PRIV, config.PROT, config.PUBL, config.OPEN)
        self.add_argument('-r', '--realm', choices=cs, help='Realm of action')
        self.add_argument('-c', '--clear', action=self.st, help='Clear before')

    def parse_args(self, args=None):
        a = super().parse_args(args)
        if a.clear:
            with start_stop_log(log, 'Deleting *.qnr and *.lock'):
                q = config.qnar_dst
                for p in (a.base / q).glob('*.qnr'):
                    p.unlink()
                for p in (a.base / q / 'filts').glob('*.qnr'):
                    p.unlink()
                for p in (a.base / q).glob('*.lock'):
                    p.unlink()
                for p in (a.base / q / 'filts').glob('*.lock'):
                    p.unlink()
        return a


class MArgs(CArgs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_argument('files', nargs='*', help='Files to read from')
        self.add_argument('-w', '--wdir', help='Path to work dir')
        self.add_argument('-p', '--pool', action=self.st, help='Pooled action')
        self.add_argument(
            '-d', '--dejunk_only', action=self.st, help='Only dejunk messages')
