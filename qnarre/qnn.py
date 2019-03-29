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

from .base import config
from .mboxes import Mboxes
from .log import Logger, start_stop_log
from .resource import resource
from .dispatch import Dispatch
# from .ptorch import TorchOne, TorchTwo
# from .tflow import Mnist

log = Logger(__name__)


class Qnn(Dispatch):

    _res_path = config.qnar_dst + 'qnn.qnr'

    _blog = 'blog'
    _ctxt = None

    @classmethod
    def globals(cls):
        return globals()

    def setup(self, **kw):
        # TorchOne().loop()
        # TorchTwo().loop()
        # Mnist().loop()
        """
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Setting up Qnn'):
                dst = '/' + self.realm
                Mboxes(self.base).export_to(dst, **kw)
        """

    def learn(self, **kw):
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Setting up Qnn'):
                dst = '/' + self.realm
                Mboxes(self.base).export_to(dst, **kw)

    def guess(self, **kw):
        with resource(self.ctxt) as ctxt:
            kw.update(ctxt=ctxt)
            with start_stop_log(log, 'Setting up Qnn'):
                dst = '/' + self.realm
                Mboxes(self.base).export_to(dst, **kw)
