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

import logging as lg
import contextlib as cl

lg.basicConfig(
    level=lg.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    # filename='/tmp/qnarre.log',
    filename="/tmp/qnarre.log",
    # filemode='w'
    filemode="w",
)  # 'a'

ch = lg.StreamHandler()
ch.setLevel(lg.WARNING)
ch.setFormatter(lg.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))

lg.getLogger().addHandler(ch)


class Logger(lg.LoggerAdapter):
    def __init__(self, name, extra=None):
        super().__init__(lg.getLogger(name), extra or {})

    def log(self, level, msg, *args, **kw):
        if self.isEnabledFor(level):
            msg, kw = self.process(msg, kw)

            class Msg:
                def __init__(self, fmt, args):
                    self.fmt = fmt
                    self.args = args

                def __str__(self):
                    return self.fmt.format(*self.args)

            self.logger._log(level, Msg(msg, args), (), **kw)


@cl.contextmanager
def start_stop_log(log, msg):
    m = msg + "..."
    log.info(m)
    print(m, end="")
    yield
    m += " done"
    log.info(m)
    print("\n" + m)


log = Logger(__name__)
